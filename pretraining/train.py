import sys
from pathlib import Path

import polars
import torch
import zarr
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import RIGRAtomFeaturizer
from chemprop.featurizers.bond import RIGRBondFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN, metrics
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from rdkit.rdBase import BlockLogs
from torch.utils.data import DataLoader

from .config import (
    EPOCHS,
    FEATURIZER,
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
)
from .dataset import ChempropChunkwiseZarrDataset
from .random_dropout_mse import RandomDropoutMSE

if __name__ == "__main__":
    # shh
    bl = BlockLogs()

    try:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    except:
        print("usage: python train.py <input_dir> <output_dir>")
        exit(1)

    training_store = input_dir / "train_rescaled.zarr"
    validation_store = input_dir / "val_rescaled.zarr"
    train_smiles_file = input_dir / "train_smiles.parquet"
    val_smiles_file = input_dir / "val_smiles.parquet"

    z = zarr.open_array(training_store, mode="r")
    n_features = z.shape[1]
    del z

    train_smiles = polars.read_parquet(train_smiles_file)["SMILES"].to_list()
    val_smiles = polars.read_parquet(val_smiles_file)["SMILES"].to_list()

    atom_featurizer = RIGRAtomFeaturizer() if FEATURIZER == "rigr" else None
    bond_featurizer = RIGRBondFeaturizer() if FEATURIZER == "rigr" else None
    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer)

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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=None, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=None, shuffle=False, num_workers=4, persistent_workers=True)

    model = MPNN(
        BondMessagePassing(
            d_v=featurizer.atom_fdim,
            d_e=featurizer.bond_fdim,
            d_h=MP_HIDDEN_SIZE,
            depth=MP_DEPTH,
            activation=MP_ACTIVATION,
        ),
        NormAggregation(),
        predictor=RegressionFFN(
            n_tasks=n_features, hidden_dim=FNN_HIDDEN_SIZE, num_layers=FNN_HIDDEN_LAYERS, activation=FNN_ACTIVATION, criterion=RandomDropoutMSE()
        ),
        metrics=[RandomDropoutMSE(), metrics.MSE(), metrics.MAE(), metrics.R2Score(), metrics.RMSE()],
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
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Reloading best model from checkpoint file: {ckpt_path}")
    model = model.__class__.load_from_checkpoint(ckpt_path, map_location="cpu")
    trainer.test(model, val_dataloader)
    torch.save(model, output_dir / "best.pt")
