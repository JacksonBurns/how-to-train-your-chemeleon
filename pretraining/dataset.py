import torch
import zarr

from rdkit.Chem import MolFromSmiles
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
import numpy as np
from chemprop.data.collate import TrainingBatch, BatchMolGraph



class ChempropChunkwiseZarrDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], zarr_store: str, featurizer: SimpleMoleculeMolGraphFeaturizer):
        self.smiles = np.array(smiles)
        self.z = zarr.open_array(zarr_store)
        assert self.z.shape[0] == len(smiles), "Mismatched smiles and feature sizes"
        self.len = self.z.nchunks
        self.n_rows = len(smiles)
        self.chunksize = self.z.chunks[0]
        self.featurizer = featurizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        start_idx = idx * self.chunksize
        stop_idx = start_idx + self.chunksize
        return TrainingBatch(
            BatchMolGraph([self.featurizer(MolFromSmiles(s)) for s in self.smiles[start_idx:stop_idx]]),
            None,
            None,
            self.z[start_idx:stop_idx, :],
            torch.ones((stop_idx - start_idx, 1)),
            None,
            None,
        )
