import os
import sys
from pathlib import Path

import numpy as np
import polars
import torch
import zarr
from chemprop.featurizers import BatchCuikMolGraph
from chemprop.nn import NormAggregation, metrics
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from rdkit.rdBase import BlockLogs
from torch.utils.data import DataLoader

from attention_atom_mp import AttentionAtomMessagePassing
from multi_task_dataset import MultiTaskChunkwiseZarrDataset
from now import NOW

from train import PatchedCuikmolmakerMolGraphFeaturizer


DROPOUT_FRACTION = 0.70


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
    """
    Computes uncertainty-weighted MSE for multiple tasks containing missing data.
    Assumes targets are already normalized (mean=0, var=1).
    
    Based on homoscedastic task uncertainty: 
    Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses..."
    Paper: https://arxiv.org/abs/1705.07115
    """

    def __init__(self, dropout_fraction: float = DROPOUT_FRACTION):
        super().__init__()
        self.dropout_fraction = dropout_fraction
        
        # Initialize log-variances (s) for each task.
        # Initializing at 0.0 means exp(-0) = 1, so the initial weighting 
        # is just 0.5 * MSE, behaving closely to an unweighted sum at step 0.
        self.log_vars = torch.nn.ParameterDict({
            "descriptor": torch.nn.Parameter(torch.tensor(0.0)),
            "fingerprint": torch.nn.Parameter(torch.tensor(0.0))
        })

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(preds.values())).device)

        for task in ("descriptor", "fingerprint"):
            p = preds[task]
            t = targets[task]
            w = weights[task]

            # 1. Safely handle missing data (NaNs) in the targets
            t_safe = torch.nan_to_num(t, nan=0.0)

            # 2. Create the random dropout mask
            dropout_mask = (torch.rand_like(p) > self.dropout_fraction).float()

            # 3. Combine the dataset validity weights and the dropout mask
            combined_mask = w * dropout_mask

            # 4. Compute the masked squared error
            squared_error = (p - t_safe) ** 2
            masked_error = squared_error * combined_mask

            # 5. Calculate the base Mean Squared Error (MSE)
            task_mse = masked_error.sum() / (combined_mask.sum() + 1e-8)

            # 6. Apply Homoscedastic Task Uncertainty Weighting
            # Formula: L = 0.5 * exp(-s) * Loss + 0.5 * s
            log_var = self.log_vars[task]
            task_weighted_loss = 0.5 * torch.exp(-log_var) * task_mse + 0.5 * log_var

            # Store metrics. It is highly recommended to track both the raw MSE 
            # and the log_vars during training to ensure they don't diverge.
            losses[f"{task}_raw_mse"] = task_mse.item()
            losses[f"{task}_weighted"] = task_weighted_loss.item()
            losses[f"{task}_log_var"] = log_var.item() 
            
            total_loss = total_loss + task_weighted_loss

        losses["total"] = total_loss.item()
        
        return total_loss, losses


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
        # force the validation loss to use the training loss's learnable parameters
        self.val_loss_fn.log_vars = self.loss_fn.log_vars

    def forward(self, graph: BatchCuikMolGraph) -> dict[str, torch.Tensor]:
        hidden = self.mp(graph)
        graph_vec = self.aggregator(hidden, graph.batch)
        return self.predictor(graph_vec)

    def _step(self, batch, split: str):
        preds = self(batch["graph"])
        loss_fn = self.loss_fn if split == "train" else self.val_loss_fn
        loss, loss_dict = loss_fn(preds, batch["targets"], batch["weights"])
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch["targets"]["fingerprint"].shape[0])
        for task, val in loss_dict.items():
            if task != "total":
                self.log(f"{split}/{task}_loss", val, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch["targets"]["fingerprint"].shape[0])
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
    FEATURIZER = "RIGR"

    try:
        desc_dir = Path(sys.argv[1])
        fp_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
    except:
        print("usage: python train_multitask.py <descriptor_dir> <fingerprint_dir> <output_dir>")
        print("")
        print("note: ensure that rows_per_chunk is the same between Zarr arrays and that SMILES are the same!")
        exit(1)

    if not desc_dir.exists():
        print(f"Error: {desc_dir} not found.")
        exit(1)

    if not fp_dir.exists():
        print(f"Error: {fp_dir} not found.")
        exit(1)

    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / NOW
    output_dir.mkdir(exist_ok=True)

    if not Path("results.csv").exists():
        with open("results.csv", "w") as f:
            f.write("run_name,val_mse\n")

    train_desc_store = desc_dir / "train_rescaled.zarr"
    val_desc_store = desc_dir / "val_rescaled.zarr"
    train_fp_store = fp_dir / "train_rescaled.zarr"
    val_fp_store = fp_dir / "val_rescaled.zarr"
    train_smiles_file = desc_dir / "train_smiles.parquet"
    val_smiles_file = desc_dir / "val_smiles.parquet"

    desc_z = zarr.open_array(train_desc_store, mode="r")
    fp_z = zarr.open_array(train_fp_store, mode="r")
    n_descriptors = desc_z.shape[1]
    n_fingerprints = fp_z.shape[1]
    rows_per_chunk = desc_z.chunks[0]
    del desc_z, fp_z

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
        d_h=32*12,  # 384
        num_heads=12,
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
        init_lr=0.00001,
        max_lr=0.0005,
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
        max_epochs=40,
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
    ckpt_path = Path(trainer.checkpoint_callback.best_model_path)

    model = MultiTaskModel.load_from_checkpoint(ckpt_path, weights_only=False)
    val_metrics = trainer.validate(model, val_dataloader, verbose=False)
    rank_zero_info(f"Best model file: {ckpt_path}")
    rank_zero_info(f"Best model validation loss: {val_metrics[0]['val/loss']:.5f}")

    if trainer.global_rank == 0:
        with open("results.csv", "a") as f:
            f.write(f"{output_dir.name},{val_metrics[0]['val/loss']:.5f}\n")
    
    torch.save({"hyper_params": dict(model.mp.hparams), "state_dict": model.mp.state_dict()}, ckpt_path.parent.resolve() / (ckpt_path.stem + "_mp.pt"))
