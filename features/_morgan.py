import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFingerprintGenerator
from rdkit.rdBase import BlockLogs
from tqdm import tqdm

from get_chunksize import get_chunk_rows

logger = logging.getLogger(__name__)

DTYPE = np.float16
RADIUS = 3
FP_SIZE = 2048

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)


def calculate(molecule: str | Mol) -> np.ndarray[DTYPE]:
    if not isinstance(molecule, Mol):
        molecule = Chem.MolFromSmiles(molecule)
    out = np.full(FP_SIZE, np.nan, dtype=DTYPE)
    if molecule is None:
        return out
    try:
        fp = fpgen.GetCountFingerprintAsNumPy(molecule)
        return np.where(fp.astype(bool), fp, out).astype(DTYPE)
    except Exception as e:
        smiles = "<invalid>"
        try:
            smiles = Chem.MolToSmiles(molecule)
        except:
            pass
        logger.warning(f"Morgan fingerprint failed for SMILES {smiles}: {repr(e)}")
        return out


def _validate_smiles(smi: str) -> str:
    """Return SMILES if valid, else empty string."""
    if smi is None:
        return ""
    try:
        return smi if Chem.MolFromSmiles(smi) is not None else ""
    except Exception:
        return ""


if __name__ == "__main__":
    import zarr

    with BlockLogs():
        try:
            in_file = Path(sys.argv[1])
            out_file = Path(sys.argv[2])
            i = int(sys.argv[3])
        except:
            print(
                f"usage: python _morgan.py </path/to/input_file.parquet> </path/to/output_file.zarr> <start_index>"
            )
            exit(1)

        df = (
            pl.read_parquet(in_file)
            .filter(pl.col("SMILES").str.len_chars() < 100)  # not needed, but for parity with osmordred
        )

        smiles = df["SMILES"].to_list()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            cleaned_smiles = list(
                tqdm(
                    ex.map(_validate_smiles, smiles, chunksize=1000),
                    total=len(smiles),
                    desc="Validating SMILES",
                )
            )

        pl.DataFrame({"SMILES": cleaned_smiles}).write_parquet(
            out_file.parent / f"{in_file.stem}_validated_smiles.parquet"
        )
        del smiles, cleaned_smiles

        n_mols = df.shape[0]
        shape = (n_mols, FP_SIZE)
        chunk_rows = get_chunk_rows(DTYPE, FP_SIZE)
        chunk_shape = (chunk_rows, FP_SIZE)

        if Path(out_file).exists():
            print(f"found previous Zarr array at {out_file}, appending")
            z = zarr.open_array(out_file)
        else:
            z = zarr.create_array(
                store=out_file,
                shape=shape,
                chunks=chunk_shape,
                dtype=DTYPE,
                compressors=None,
                fill_value=np.nan,
            )

        p = Pool(64)
        with tqdm(
            total=n_mols,
            initial=i,
            desc=f"Calculating Features (batch size {chunk_rows})",
            file=sys.stdout,
        ) as pbar:
            while i < n_mols:
                z[i : i + chunk_rows, :] = np.stack(
                    p.map(calculate, df["SMILES"][i : i + chunk_rows])
                )
                pbar.update(chunk_rows)
                i += chunk_rows
            z[i - chunk_rows : n_mols, :] = np.stack(
                p.map(calculate, df["SMILES"][i - chunk_rows : n_mols])
            )
            pbar.update(n_mols - (i - chunk_rows))
