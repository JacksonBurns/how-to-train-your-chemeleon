import math
import numpy as np
import torch
import zarr
from config import CHUNKS_PER_BATCH


class MultiTaskChunkwiseZarrDataset(torch.utils.data.Dataset):
    """Dataset that loads both continuous descriptors and morgan fingerprints.
    
    Returns a dict per item:
      - graph: BatchCuikMolGraph from the featurizer
      - targets: dict with 'descriptor' and 'fingerprint' tensors (each shape [batch, n_features])
      - weights: dict with 'descriptor' and 'fingerprint' weight tensors (each shape [batch, 1])
    """

    def __init__(
        self,
        smiles: list[str],
        descriptor_zarr: str,
        fingerprint_zarr: str,
        featurizer,
    ):
        self.smiles = np.array(smiles)
        self.descriptor_zarr = descriptor_zarr
        self.fingerprint_zarr = fingerprint_zarr
        self.descriptor_z = None
        self.fingerprint_z = None
        self.featurizer = featurizer

        # Open temporarily just to get metadata
        _d = zarr.open_array(descriptor_zarr)
        _f = zarr.open_array(fingerprint_zarr)
        assert _d.shape[0] == len(smiles), "Mismatched smiles and descriptor sizes"
        assert _f.shape[0] == len(smiles), "Mismatched smiles and fingerprint sizes"
        assert _d.chunks[0] == _f.chunks[0], "Descriptor and fingerprint chunk sizes differ"

        self.n_rows = len(smiles)
        self.chunksize = _d.chunks[0]
        self.n_descriptors = _d.shape[1]
        self.n_fingerprints = _f.shape[1]

        self.items_per_batch = self.chunksize * CHUNKS_PER_BATCH
        self.len = math.ceil(self.n_rows / self.items_per_batch)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        if self.descriptor_z is None:
            self.descriptor_z = zarr.open_array(self.descriptor_zarr)
        if self.fingerprint_z is None:
            self.fingerprint_z = zarr.open_array(self.fingerprint_zarr)

        start_idx = idx * self.items_per_batch
        stop_idx = min(start_idx + self.items_per_batch, self.n_rows)
        batch_smiles = self.smiles[start_idx:stop_idx].tolist()

        descriptor_targets = torch.tensor(
            self.descriptor_z[start_idx:stop_idx, :], dtype=torch.float32
        )
        fingerprint_targets = torch.tensor(
            self.fingerprint_z[start_idx:stop_idx, :], dtype=torch.float32
        )
        descriptor_weights = torch.ones(
            (descriptor_targets.shape[0], 1), dtype=torch.float32
        )
        fingerprint_weights = torch.ones(
            (fingerprint_targets.shape[0], 1), dtype=torch.float32
        )

        try:
            graph = self.featurizer(batch_smiles)
        except Exception as e:
            with open("batch_errors.log", "a") as f:
                f.write(
                    f"Error processing batch {idx} on rank {torch.distributed.get_rank()} "
                    f"(rows {start_idx}-{stop_idx}): {e}\n"
                )
            graph = self.featurizer([""] * (stop_idx - start_idx))
            descriptor_targets = torch.zeros_like(descriptor_targets)
            fingerprint_targets = torch.zeros_like(fingerprint_targets)

        return {
            "graph": graph,
            "targets": {
                "descriptor": descriptor_targets,
                "fingerprint": fingerprint_targets,
            },
            "weights": {
                "descriptor": descriptor_weights,
                "fingerprint": fingerprint_weights,
            },
        }
