import os
import sys
from pathlib import Path

import cuik_molmaker
import numpy as np
import polars
import torch
import zarr
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.featurizers import BatchCuikMolGraph, CuikmolmakerMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import Aggregation, AggregationRegistry, RegressionFFN, metrics
from chemprop.nn.message_passing.base import _BondMessagePassingMixin, _MessagePassingBase
from chemprop.nn.metrics import MSE, LossFunctionRegistry, MetricRegistry
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from rdkit.rdBase import BlockLogs
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import ChempropChunkwiseZarrDataset
from now import NOW
from config import CHUNKS_PER_BATCH


DROPOUT_FRACTION = 0.30
FEATURIZER = "RIGR"  # one of: "V2", "RIGR"


@LossFunctionRegistry.register("rdmse")
@MetricRegistry.register("rdmse")
class RandomDropoutMSE(MSE):
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        lt_mask: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> None:
        # overrides parent to generate a randomly initialized mask
        random_mask = (torch.rand_like(targets) > DROPOUT_FRACTION).bool()
        mask = (
            random_mask if mask is None else torch.logical_and(random_mask, mask)
        )  # i.e., include if both masks requests so
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation:

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v
    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="mean", include_self=False
        )


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


class MultiweightMessagePassing(_BondMessagePassingMixin, _MessagePassingBase):
    r"""A variant of BondMessagePassing where the hidden weight matrix (W_h)
    is untied across message passing steps (depth).

    Instead of reapplying the same matrix, a distinct W_h_i is learned for each iteration.
    """

    def __init__(self, *args, **kwargs):
        # 1. Run the base initialization, which will temporarily create a single W_h
        super().__init__(*args, **kwargs)

        # 2. Extract dimensions and bias from the temporarily created matrix
        d_h = self.W_h.in_features
        bias = self.W_h.bias is not None

        # 3. Overwrite W_h with a ModuleList of untied matrices.
        # The message passing loop runs (depth - 1) times, so we need (depth - 1) matrices.
        self.W_h = nn.ModuleList([nn.Linear(d_h, d_h, bias=bias) for _ in range(self.depth - 1)])

        # LayerNorms for regularization
        self.norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(self.depth - 1)])

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ):
        # Standard setup required by the base class.
        # The single W_h returned here is immediately overwritten by our __init__ above.
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        return W_i, W_h, W_o, W_d

    def update(self, M_t: Tensor, H_0: Tensor, step: int) -> Tensor:
        """Calculate the updated hidden state using the step-specific weight matrix"""
        # Select the specific layernorm/weight matrix for this depth iteration
        M_norm = self.norms[step](M_t)
        H_t = self.W_h[step](M_norm)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)

        # We replace the `for _ in range(1, self.depth)` with an enumerated loop
        # so we can pass the step index (0 to depth-2) to the update function
        for step in range(self.depth - 1):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0, step)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return self.finalize(M, bmg.V, V_d)


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
    
    if not Path("results.csv").exists():
        with open("results.csv", "w") as f:
            f.write("run_name,val_mse\n")

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
    
    mp = MultiweightMessagePassing(
            d_v=featurizer.atom_fdim,
            d_e=featurizer.bond_fdim,
            d_h=2_048,
            depth=4,
            activation=torch.nn.ReLU(),
    )

    model = MPNN(
        mp,
        MeanAggregation(),
        predictor=RegressionFFN(
            n_tasks=n_features,
            input_dim=mp.output_dim,
            hidden_dim=2_048,
            n_layers=2,
            activation=torch.nn.ReLU(),
            criterion=RandomDropoutMSE(),
        ),
        metrics=[metrics.MSE(), metrics.MAE(), metrics.R2Score(), metrics.RMSE()],
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
            monitor="val/mse",
            mode="min",
            verbose=False,
            patience=1,
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
        max_epochs=10,
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
    # get the validation performance
    model = MPNN.load_from_checkpoint(ckpt_path)
    val_metrics = trainer.validate(model, val_dataloader, verbose=False)
    rank_zero_info(f"Best model file: {ckpt_path}")
    rank_zero_info(f"Best model validation mse: {val_metrics[0]['val/mse']:.5f}")
    # write on rank zero only
    if trainer.global_rank == 0:
        with open("results.csv", "a") as f:
            f.write(f"{output_dir.name},{val_metrics[0]['val/mse']:.5f}\n")
