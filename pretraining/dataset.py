import math

import numpy as np
import torch
import zarr
from chemprop.data.collate import TrainingBatch
from chemprop.featurizers import CuikmolmakerMolGraphFeaturizer

from config import CHUNKS_PER_BATCH


class ChempropChunkwiseZarrDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], zarr_store: str, featurizer: CuikmolmakerMolGraphFeaturizer):
        self.smiles = np.array(smiles)
        self.zarr_store = zarr_store
        self.z = None  # will be lazily loaded on first access
        _z = zarr.open_array(zarr_store)
        assert _z.shape[0] == len(smiles), "Mismatched smiles and feature sizes"

        self.n_rows = len(smiles)
        self.chunksize = _z.chunks[0]
        
        del _z

        # Calculate the effective size of a batch (multiple chunks)
        self.items_per_batch = self.chunksize * CHUNKS_PER_BATCH

        # Update length to reflect the new number of multi-chunk groups
        self.len = math.ceil(self.n_rows / self.items_per_batch)
        self.featurizer = featurizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        # Lazily open the zarr array only when the first item is requested by a worker.
        if self.z is None:
            self.z = zarr.open_array(self.zarr_store)

        # Calculate start and stop indices based on the combined batch size
        start_idx = idx * self.items_per_batch
        stop_idx = min(start_idx + self.items_per_batch, self.n_rows)

        # Chemprop's normal featurizer returns empty features for empty SMILES
        # whereas cuik-molmaker skips them entirely, causing batch size to be wrong.
        # so, we must filter out the rows corresponding to invalid SMILES
        smiles_batch = self.smiles[start_idx:stop_idx]
        valid_mask = (smiles_batch != "")
        features = self.featurizer(smiles_batch[valid_mask].tolist())
        targets = torch.tensor(self.z[start_idx:stop_idx, :], dtype=torch.float32)[valid_mask]
        weights = torch.ones((targets.shape[0], 1), dtype=torch.float32)

        return TrainingBatch(
            features,
            None,
            None,
            targets,
            weights,
            None,
            None,
        )