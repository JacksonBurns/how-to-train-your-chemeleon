import os
import sys
from pathlib import Path

import cuik_molmaker
import numpy as np
import polars
import torch
import zarr
from chemprop.featurizers import BatchCuikMolGraph, CuikmolmakerMolGraphFeaturizer
from chemprop.nn import NormAggregation, metrics
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from rdkit.rdBase import BlockLogs
from torch.utils.data import DataLoader

from config import CHUNKS_PER_BATCH
from attention_atom_mp import AttentionAtomMessagePassing
from multi_task_dataset import MultiTaskChunkwiseZarrDataset
from now import NOW


DROPOUT_FRACTION = 0.70
FEATURIZER = "RIGR"


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

            dummy_atoms = np.zeros(
                (len(empty_indices_arr), atom_feats.shape[1]), dtype=atom_feats.dtype
            )
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


class MultiTaskPredictor(torch.nn.Module):
    """Two-head predictor: continuous descriptors + morgan fingerprints."""

    def __init__(self, input_dim: int, hidden_dim: int, n_descriptors: int, n_fingerprints: int):
        super().__init__()
        self.descriptor_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_descriptors),
        )
        self.fingerprint_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_fingerprints),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "descriptor": self.descriptor_head(x),
            "fingerprint": self.fingerprint_head(x),
        }


class MultiTaskMSE(torch.nn.Module):
    """Combined MSE loss with random dropout and magnitude balancing.
    
    Divides each task's loss by its output dimensionality so that the two
    losses are of similar magnitude regardless of descriptor/fingerprint count.
    """

    def __init__(self, dropout_fraction: float = DROPOUT_FRACTION):
        super().__init__()
        self.dropout_fraction = dropout_fraction

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        losses = {}
        for task in ("descriptor", "fingerprint"):
            p = preds[task]
            t = targets[task]
            w = weights[task]
            mask = (torch.rand_like(p) > self.dropout_fraction).bool()
            masked = (p - t) ** 2
            weighted = masked * w.squeeze(-1)
            task_loss = weighted.sum() / (mask.sum() + 1e-8)
            # Normalize by output dimensionality for magnitude balancing
            task_loss = task_loss / p.shape[1]
            losses[task] = task_loss.item()
        total = losses["descriptor"] + losses["fingerprint"]
        losses["total"] = total.item()
        return torch.tensor(total, device=next(iter(preds.values())).device), losses


class MultiTaskModel(LightningModule):
    def __init__(
        self,
        mp: AttentionAtomMessagePassing,
        predictor: MultiTaskPredictor,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mp = mp
        self.aggregator = NormAggregation()
        self.predictor = predictor
        self.loss_fn = MultiTaskMSE()
        self.val_loss_fn = MultiTaskMSE(dropout_fraction=0.0)
        self.metrics = {
            "descriptor": metrics.MSE(),
            "fingerprint": metrics.MSE(),
        }

    def forward(self, graph: BatchCuikMolGraph) -> dict[str, torch.Tensor]:
        hidden = self.mp(graph)
        graph_vec = self.aggregator(hidden, graph.batch)
        return self.predictor(graph_vec)

    def _step(self, batch, split: str):
        preds = self(batch["graph"])
        loss_fn = self.loss_fn if split == "train" else self.val_loss_fn
        loss, loss_dict = loss_fn(preds, batch["targets"], batch["weights"])
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        for task, val in loss_dict.items():
            if task != "total":
                self.log(f"{split}/{task}_loss", val, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.init_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_epochs / self.trainer.max_epochs,
            final_div_factor=self.hparams.max_lr / self.hparams.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == "__main__":
    bl = BlockLogs()

    try:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    except:
        print("usage: python train_multitask.py <input_dir> <output_dir>")
        exit(1)

    if not input_dir.exists():
        print(f"Error: {input_dir} not found.")
        exit(1)

    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / NOW
    output_dir.mkdir(exist_ok=True)

    if not Path("results.csv").exists():
        with open("results.csv", "w") as f:
            f.write("run_name,val_mse\n")

    train_desc_store = input_dir / "train_descriptor_rescaled.zarr"
    val_desc_store = input_dir / "val_descriptor_rescaled.zarr"
    train_fp_store = input_dir / "train_fingerprint_rescaled.zarr"
    val_fp_store = input_dir / "val_fingerprint_rescaled.zarr"
    train_smiles_file = input_dir / "train_smiles.parquet"
    val_smiles_file = input_dir / "val_smiles.parquet"

    desc_z = zarr.open_array(train_desc_store, mode="r")
    fp_z = zarr.open_array(train_fp_store, mode="r")
    n_descriptors = desc_z.shape[1]
    n_fingerprints = fp_z.shape[1]
    rows_per_chunk = desc_z.chunks[0]
    del desc_z, fp_z

    bytes_per_row = n_descriptors * 2
    target_rows_for_1gb = (1024**3) // bytes_per_row
    shard_multiplier = max(1, round(target_rows_for_1gb / rows_per_chunk))

    train_smiles = polars.read_parquet(train_smiles_file)["SMILES"].to_list()
    val_smiles = polars.read_parquet(val_smiles_file)["SMILES"].to_list()

    featurizer = PatchedCuikmolmakerMolGraphFeaturizer(FEATURIZER)

    train_dataset = MultiTaskChunkwiseZarrDataset(
        train_smiles, train_desc_store, train_fp_store, featurizer
    )
    val_dataset = MultiTaskChunkwiseZarrDataset(
        val_smiles, val_desc_store, val_fp_store, featurizer
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=None, num_workers=2, persistent_workers=True
    )

    mp = AttentionAtomMessagePassing(
        d_v=featurizer.atom_fdim,
        d_e=featurizer.bond_fdim,
        d_h=512,
        num_heads=4,
        num_layers=6,
        tied_weights=True,
        gate=True,
    )

    predictor = MultiTaskPredictor(
        input_dim=mp.output_dim,
        hidden_dim=mp.d_h,
        n_descriptors=n_descriptors,
        n_fingerprints=n_fingerprints,
    )

    model = MultiTaskModel(
        mp,
        predictor,
        init_lr=0.0001,
        max_lr=0.001,
        final_lr=0.0001,
        warmup_epochs=2,
    )
    rank_zero_info(model)

    tensorboard_logger = TensorBoardLogger(
        output_dir,
        name="tensorboard_logs",
        default_hp_metric=False,
    )
    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            verbose=False,
            patience=2,
        ),
        ModelCheckpoint(
            monitor="val/loss",
            save_top_k=2,
            mode="min",
            dirpath=output_dir / "checkpoints",
        ),
    ]
    callbacks[1].STARTING_VERSION = 0
    trainer = Trainer(
        max_epochs=20,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        val_check_interval=0.5,
    )
    restart_ckpt = os.environ.get("RESTART_CKPT", None)
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=restart_ckpt,
        weights_only=restart_ckpt is None,
    )
    ckpt_path = trainer.checkpoint_callback.best_model_path

    model = MultiTaskModel.load_from_checkpoint(ckpt_path)
    val_metrics = trainer.validate(model, val_dataloader, verbose=False)
    rank_zero_info(f"Best model file: {ckpt_path}")
    rank_zero_info(f"Best model validation loss: {val_metrics[0]['val/loss']:.5f}")

    if trainer.global_rank == 0:
        with open("results.csv", "a") as f:
            f.write(f"{output_dir.name},{val_metrics[0]['val/loss']:.5f}\n")
