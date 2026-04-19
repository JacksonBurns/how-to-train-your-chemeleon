"""
features.py

Streaming Mordred descriptor computation:
- ProcessPoolExecutor
- threadpoolctl limits BLAS threads to 1
- End-to-end chunk processing in workers
- Single pass, no buffering
- Invalid SMILES leave NaN rows in Zarr
"""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from tqdm import tqdm

from threadpoolctl import threadpool_limits

from rdkit import rdBase
from rdkit.Chem import MolFromSmiles, RemoveHs
from mordred import Calculator, descriptors

from .get_chunksize import get_chunk_rows

warnings.filterwarnings("ignore", category=FutureWarning)


def process_chunk(start_idx, smiles_chunk, n_features):
    """
    Processes a chunk and returns:
    (start_idx, feature_block)

    feature_block shape = (len(chunk), n_features)
    Invalid rows are left as NaN.
    """

    with threadpool_limits(limits=1):
        calc = Calculator(descriptors, ignore_3D=True)
        calc.config(timeout=1)

        n = len(smiles_chunk)
        feats = np.full((n, n_features), np.nan, dtype=np.float32)

        for i, smi in enumerate(smiles_chunk):
            # cheap filters first
            if len(smi) >= 150 or "." in smi:
                continue

            mol = MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                RemoveHs(mol, updateExplicitCount=True)
            except Exception:
                continue

            mol.SetProp("_Name", "")

            try:
                row = (
                    calc.pandas([mol], quiet=True, nproc=1)
                    .fill_missing()
                    .to_numpy(dtype=np.float32)
                )
                feats[i, :] = row[0]
            except Exception:
                # leave as NaN on any failure
                continue

        return start_idx, feats


if __name__ == "__main__":
    blocker = rdBase.BlockLogs()

    try:
        smiles_file = sys.argv[1]
        out_file = sys.argv[2]
    except Exception:
        print("Usage: python _mordred.py SMILES_FILE OUTPUT_PATH")
        sys.exit(1)

    # Read SMILES
    with open(smiles_file, "r") as f:
        smiles = [line.strip() for line in tqdm(f, desc="Reading SMILES")]

    n_mols = len(smiles)

    # Precompute feature dimension
    calc = Calculator(descriptors, ignore_3D=True)
    n_features = len(calc)

    # Chunking
    dtype = np.float32
    chunk_rows = get_chunk_rows(dtype, n_features)
    print(f"Rows per chunk: {chunk_rows}")

    z = zarr.create_array(
        store=out_file,
        shape=(n_mols, n_features),
        chunks=(chunk_rows, n_features),
        dtype=dtype,
        compressors=None,
        fill_value=np.nan,
    )

    # Submit chunks
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = []

        for start in range(0, n_mols, chunk_rows):
            chunk = smiles[start : start + chunk_rows]
            futures.append(
                executor.submit(process_chunk, start, chunk, n_features)
            )

        # Consume as they complete (out-of-order OK, they are written to correct location by start_idx)
        with tqdm(total=n_mols, desc="Calculating features") as pbar:
            for fut in as_completed(futures):
                start_idx, feats = fut.result()
                n = feats.shape[0]

                z[start_idx : start_idx + n, :] = feats
                pbar.update(n)
