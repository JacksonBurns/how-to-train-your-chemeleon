import math
import torch
import zarr
import numpy as np

from rdkit.Chem import MolFromSmiles
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.collate import TrainingBatch, BatchMolGraph

from .config import CHUNKS_PER_BATCH

class ChempropChunkwiseZarrDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], zarr_store: str, featurizer: SimpleMoleculeMolGraphFeaturizer):
        self.smiles = np.array(smiles)
        self.z = zarr.open_array(zarr_store)
        assert self.z.shape[0] == len(smiles), "Mismatched smiles and feature sizes"
        
        self.n_rows = len(smiles)
        self.chunksize = self.z.chunks[0]
        
        # Calculate the effective size of a batch (multiple chunks)
        self.items_per_batch = self.chunksize * CHUNKS_PER_BATCH
        
        # Update length to reflect the new number of multi-chunk groups
        self.len = math.ceil(self.n_rows / self.items_per_batch)
        self.featurizer = featurizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        # Calculate start and stop indices based on the combined batch size
        start_idx = idx * self.items_per_batch
        stop_idx = min(start_idx + self.items_per_batch, self.n_rows)
        
        # Zarr handles cross-chunk slicing automatically.
        # Wrap the numpy array return in a torch tensor to ensure Lightning compatibility.
        targets = torch.tensor(
            self.z[start_idx:stop_idx, :], 
            dtype=torch.float32
        )
        weights = torch.ones((stop_idx - start_idx, 1), dtype=torch.float32)

        return TrainingBatch(
            BatchMolGraph([self.featurizer(MolFromSmiles(s)) for s in self.smiles[start_idx:stop_idx]]),
            None,
            None,
            targets,
            weights,
            None,
            None,
        )
