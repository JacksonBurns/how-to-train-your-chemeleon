import math
import numpy as np
import torch
import zarr
from chemprop.data.collate import TrainingBatch

from config import CHUNKS_PER_BATCH

class ChempropChunkwiseZarrDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], zarr_store: str, featurizer: "PatchedCuikmolmakerMolGraphFeaturizer"):
        self.smiles = np.array(smiles)
        self.zarr_store = zarr_store
        self.z = None  # Lazily loaded to prevent multiprocessing hangs
        
        # Open temporarily just to get metadata
        _temp_z = zarr.open_array(zarr_store)
        assert _temp_z.shape[0] == len(smiles), "Mismatched smiles and feature sizes"

        self.n_rows = len(smiles)
        self.chunksize = _temp_z.chunks[0]

        self.items_per_batch = self.chunksize * CHUNKS_PER_BATCH
        self.len = math.ceil(self.n_rows / self.items_per_batch)
        self.featurizer = featurizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        if self.z is None:
            self.z = zarr.open_array(self.zarr_store)

        start_idx = idx * self.items_per_batch
        stop_idx = min(start_idx + self.items_per_batch, self.n_rows)

        features = self.featurizer(self.smiles[start_idx:stop_idx].tolist())
        
        # Directly slice targets. The dimensions are now guaranteed to match!
        targets = torch.tensor(self.z[start_idx:stop_idx, :], dtype=torch.float32)
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
