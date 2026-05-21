import sys
import os
from pathlib import Path

import polars
import torch
import zarr
import numpy as np
from chemprop.featurizers import CuikmolmakerMolGraphFeaturizer, BatchCuikMolGraph
from chemprop.models import MPNN
from chemprop.nn import MeanAggregation, RegressionFFN, metrics, BondMessagePassing
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from rdkit.rdBase import BlockLogs
from torch.utils.data import DataLoader
import cuik_molmaker

from config import (
    EPOCHS,
    FEATURIZER,
    LR_MULTIPLIER,
    FINAL_LEARNING_RATE,
    FNN_ACTIVATION,
    FNN_HIDDEN_LAYERS,
    FNN_HIDDEN_SIZE,
    INITIAL_LEARNING_RATE,
    MAXIMUM_LEARNING_RATE,
    MP_ACTIVATION,
    MP_DEPTH,
    MP_HIDDEN_SIZE,
    PATIENCE,
    WARMUP_EPOCHS,
    CHUNKS_PER_BATCH,
    MP_TYPE,
    CV_PERCENTILE_THRESHOLD,
    COUNT_PERCENTILE_THRESHOLD,
)
from dataset import ChempropChunkwiseZarrDataset
from random_dropout_mse import RandomDropoutMSE
from multiweight_message_passing import MultiweightMessagePassing
from now import NOW

class PatchedCuikmolmakerMolGraphFeaturizer(CuikmolmakerMolGraphFeaturizer):
    def __call__(
        self,
        smiles_list: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> BatchCuikMolGraph:
        offset_carbon, duplicate_edges, add_self_loop = False, True, False

        (
            atom_feats,
            bond_feats,
            edge_index,
            rev_edge_index,
            batch,
        ) = cuik_molmaker.batch_mol_featurizer(
            smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.add_h,
            offset_carbon,
            duplicate_edges,
            add_self_loop,
        )

        empty_indices = [i for i, s in enumerate(smiles_list) if s == ""]
        if empty_indices:
            empty_indices_arr = np.array(empty_indices, dtype=np.int64)

            if edge_index.size > 0:
                shifts = np.searchsorted(empty_indices_arr, batch[edge_index], side="right")
                edge_index += shifts

            insert_positions = np.searchsorted(batch, empty_indices_arr)

            dummy_atoms = np.zeros((len(empty_indices_arr), atom_feats.shape[1]), dtype=atom_feats.dtype)
            atom_feats = np.insert(atom_feats, insert_positions, dummy_atoms, axis=0)

            batch = np.insert(batch, insert_positions, empty_indices_arr)

        atom_feats = torch.from_numpy(atom_feats)
        bond_feats = torch.from_numpy(bond_feats)
        edge_index = torch.from_numpy(edge_index)
        rev_edge_index = torch.from_numpy(rev_edge_index)
        batch = torch.from_numpy(batch)

        if atom_features_extra is not None:
            atom_features_extra = torch.tensor(atom_features_extra, dtype=torch.float32)
            atom_feats = torch.cat((atom_feats, atom_features_extra), dim=1)
        if bond_features_extra is not None:
            bond_features_extra = np.repeat(bond_features_extra, repeats=2, axis=0)
            bond_features_extra = torch.tensor(bond_features_extra, dtype=torch.float32)
            bond_feats = torch.cat((bond_feats, bond_features_extra), dim=1)

        return BatchCuikMolGraph(
            V=atom_feats,
            E=bond_feats,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            batch=batch,
        )


if __name__ == "__main__":
    # shh
    bl = BlockLogs()

    try:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    except:
        print("usage: python train.py <input_dir> <output_dir>")
        exit(1)

    if not input_dir.exists():
        print(f"Error: {input_dir} not found.")
        exit(1)

    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / NOW
    output_dir.mkdir(exist_ok=True)
    
    # dump the input args and the config into this output directory for posterity
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"input_dir: {input_dir}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"FEATURIZER: {FEATURIZER}\n")
        f.write(f"FINAL_LEARNING_RATE: {FINAL_LEARNING_RATE}\n")
        f.write(f"FNN_ACTIVATION: {FNN_ACTIVATION}\n")
        f.write(f"FNN_HIDDEN_LAYERS: {FNN_HIDDEN_LAYERS}\n")
        f.write(f"FNN_HIDDEN_SIZE: {FNN_HIDDEN_SIZE}\n")
        f.write(f"INITIAL_LEARNING_RATE: {INITIAL_LEARNING_RATE}\n")
        f.write(f"MAXIMUM_LEARNING_RATE: {MAXIMUM_LEARNING_RATE}\n")
        f.write(f"LR_MULTIPLIER: {LR_MULTIPLIER}\n")
        f.write(f"MP_ACTIVATION: {MP_ACTIVATION}\n")
        f.write(f"MP_DEPTH: {MP_DEPTH}\n")
        f.write(f"MP_HIDDEN_SIZE: {MP_HIDDEN_SIZE}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")
        f.write(f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n")
        f.write(f"MP_TYPE: {MP_TYPE}\n")
        f.write(f"CV_PERCENTILE_THRESHOLD: {CV_PERCENTILE_THRESHOLD}\n")
        f.write(f"COUNT_PERCENTILE_THRESHOLD: {COUNT_PERCENTILE_THRESHOLD}\n")

    training_store = input_dir / "train_rescaled.zarr"
    validation_store = input_dir / "val_rescaled.zarr"
    train_smiles_file = input_dir / "train_smiles.parquet"
    val_smiles_file = input_dir / "val_smiles.parquet"

    z = zarr.open_array(training_store, mode="r")
    n_features = z.shape[1]
    
    rows_per_chunk = z.chunks[0]
    bytes_per_row = n_features * 2
    target_rows_for_1gb = (1024**3) // bytes_per_row
    shard_multiplier = max(1, round(target_rows_for_1gb / rows_per_chunk))
    batches_per_shard = max(1, shard_multiplier // CHUNKS_PER_BATCH)
    
    del z

    # Load baseline precomputed statistics
    try:
        means_file = next(input_dir.glob("feature_train_means_*.npy"))
        stds_file = next(input_dir.glob("feature_train_stds_*.npy"))
        counts_file = next(input_dir.glob("feature_train_counts_*.npy"))
        
        train_means = np.load(means_file)
        train_stds = np.load(stds_file)
        train_counts = np.load(counts_file)
        
        # drop totally boring
        valid_stds_mask = train_stds > 1e-6
        rank_zero_info(f"Initial feature filtering based on std > 1e-6: Keeping {valid_stds_mask.sum()}/{n_features} features.")
        
        # Calculate the Coefficient of Variation (CV = std / mean) to account for differing scales
        # Avoid division-by-zero errors for completely zero columns by adding epsilon
        eps = 1e-8
        coef_of_variation = np.abs(train_stds / (np.abs(train_means) + eps))
        
        # Determine the cutoff value for the lowest n% percentile
        cv_cutoff_percentile = CV_PERCENTILE_THRESHOLD * 100
        cv_cutoff = np.nanpercentile(coef_of_variation, cv_cutoff_percentile)
        count_cutoff_percentile = COUNT_PERCENTILE_THRESHOLD * 100
        count_cutoff = np.nanpercentile(train_counts, count_cutoff_percentile)

        # Keep it ONLY if it is not constant AND it is not sparse noise AND it has enough counts
        valid_columns_mask = valid_stds_mask & (coef_of_variation <= cv_cutoff) & (train_counts >= count_cutoff)
        # Construct task weights tensor shape: (n_tasks,) where 1=keep, 0=drop
        task_weights_tensor = torch.from_numpy(valid_columns_mask.astype(np.float32))
        
        rank_zero_info(f"Percentile Filter Applied (Keeping top {(1 - CV_PERCENTILE_THRESHOLD):.0%}): {int(np.sum(valid_columns_mask))}/{n_features} features active.")
        rank_zero_info(f"Cutoffs -> Max CV: {cv_cutoff:.6f}, Min Finite Counts: {count_cutoff:.1f}. Dropped {int(np.sum(~valid_columns_mask))} columns.")
    except Exception as e:
        rank_zero_info(f"Warning: Processing statistics files failed ({repr(e)}).")
        exit(1)

    train_smiles = polars.read_parquet(train_smiles_file)["SMILES"].to_list()
    val_smiles = polars.read_parquet(val_smiles_file)["SMILES"].to_list()

    featurizer = PatchedCuikmolmakerMolGraphFeaturizer(FEATURIZER)

    train_dataset = ChempropChunkwiseZarrDataset(
        train_smiles,
        training_store,
        featurizer,
    )
    val_dataset = ChempropChunkwiseZarrDataset(
        val_smiles,
        validation_store,
        featurizer,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=None, shuffle=True, num_workers=2, persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=None, num_workers=2, persistent_workers=True)

    mp = BondMessagePassing if MP_TYPE == "DEFAULT" else  MultiweightMessagePassing
    model = MPNN(
        mp(
            d_v=featurizer.atom_fdim,
            d_e=featurizer.bond_fdim,
            d_h=MP_HIDDEN_SIZE,
            depth=MP_DEPTH,
            activation=MP_ACTIVATION,
        ),
        MeanAggregation(),
        predictor=RegressionFFN(
            n_tasks=n_features, input_dim=MP_HIDDEN_SIZE, hidden_dim=FNN_HIDDEN_SIZE, n_layers=FNN_HIDDEN_LAYERS, activation=FNN_ACTIVATION, criterion=RandomDropoutMSE(task_weights=task_weights_tensor)
        ),
        batch_norm=True,
        metrics=[metrics.MSE(task_weights=task_weights_tensor), metrics.MAE(task_weights=task_weights_tensor), metrics.R2Score(task_weights=task_weights_tensor), metrics.RMSE(task_weights=task_weights_tensor)],
        init_lr=INITIAL_LEARNING_RATE,
        max_lr=MAXIMUM_LEARNING_RATE,
        final_lr=FINAL_LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
    )
    rank_zero_info(model)

    tensorboard_logger = TensorBoardLogger(
        output_dir,
        name="tensorboard_logs",
        default_hp_metric=False,
    )
    callbacks = [
        EarlyStopping(
            monitor="val/mse",
            mode="min",
            verbose=False,
            patience=PATIENCE,
        ),
        ModelCheckpoint(
            monitor="val/mse",
            save_top_k=2,
            mode="min",
            dirpath=output_dir / "checkpoints",
        ),
    ]
    callbacks[1].STARTING_VERSION = 0
    trainer = Trainer(
        max_epochs=EPOCHS,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        val_check_interval=0.5,
    )
    restart_ckpt = os.environ.get("RESTART_CKPT", None)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=restart_ckpt, weights_only=restart_ckpt is None)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    # get the validation performance
    model = MPNN.load_from_checkpoint(ckpt_path)
    val_metrics = trainer.validate(model, val_dataloader, verbose=False)
    rank_zero_info(f"Best model validation metrics: {val_metrics}")
    rank_zero_info(f"Best model file: {ckpt_path}")
