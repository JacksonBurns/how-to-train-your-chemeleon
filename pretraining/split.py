import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import polars
import zarr
from tqdm import tqdm

from config import WINSORIZATION_FACTOR


def combine_stats(stat_a, stat_b):
    """
    Merges two sets of Welford statistics (n, mean, M2) into one.
    """
    na, mean_a, m2_a = stat_a
    nb, mean_b, m2_b = stat_b

    n_combined = na + nb
    delta = mean_b - mean_a

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_combined = mean_a + delta * (nb / n_combined)
        m2_combined = m2_a + m2_b + (delta**2) * (na * nb / n_combined)

    mask_a_only = (na > 0) & (nb == 0)
    mask_b_only = (nb > 0) & (na == 0)

    mean_combined[mask_a_only] = mean_a[mask_a_only]
    m2_combined[mask_a_only] = m2_a[mask_a_only]

    mean_combined[mask_b_only] = mean_b[mask_b_only]
    m2_combined[mask_b_only] = m2_b[mask_b_only]

    mean_combined = np.nan_to_num(mean_combined, nan=0.0)
    m2_combined = np.nan_to_num(m2_combined, nan=0.0)

    return n_combined, mean_combined, m2_combined


def compute_chunk_stats(args):
    """
    Worker function to compute stats directly from the source Zarr.
    """
    zarr_path, chunk_idx, chunk_rows = args

    z_array = zarr.open(zarr_path, mode="r")
    n_rows, n_cols = z_array.shape

    start = chunk_idx * chunk_rows
    end = min(start + chunk_rows, n_rows)

    # Read chunk and upcast to float64
    chunk = z_array[start:end].astype(np.float64, copy=False)

    finite = np.isfinite(chunk)
    bcount = finite.sum(axis=0)

    if not np.any(bcount):
        return (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))

    chunk_sum = np.where(finite, chunk, 0.0).sum(axis=0)
    mean = np.zeros_like(chunk_sum, dtype=np.float64)
    valid = bcount > 0
    mean[valid] = chunk_sum[valid] / bcount[valid]

    diff = np.where(finite, chunk - mean, 0.0)
    m2 = (diff * diff).sum(axis=0)

    return bcount, mean, m2


def mean_std_zarr_parallel(zarr_path, train_chunks, max_workers=None):
    """
    Computes mean and std using ONLY the training indices from the source array.
    """
    zarr_array = zarr.open(zarr_path, mode="r")
    n_cols = zarr_array.shape[1]
    chunk_rows = zarr_array.chunks[0]

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    # Prepare tasks ONLY for training chunks
    tasks = [(zarr_path, idx, chunk_rows) for idx in train_chunks]

    print(f"Calculating stats across {len(tasks)} training chunks with {max_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_chunk_stats, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing stats"):
            results.append(future.result())

    total_stats = (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))

    for part_stats in results:
        total_stats = combine_stats(total_stats, part_stats)

    final_count, final_mean, final_m2 = total_stats

    variance = np.full(n_cols, np.nan, dtype=np.float64)
    valid_final = final_count > 1

    variance[valid_final] = final_m2[valid_final] / (final_count[valid_final] - 1)
    std = np.sqrt(variance)

    return final_mean, std, final_count


def process_and_write_chunk(args):
    """ 
    Worker function to read from source, winsorize, rescale, and write to output 
    in a single pass.
    """
    in_path, out_path, in_chunk_idx, out_chunk_idx, chunk_rows, lower_limits, upper_limits, mean, std = args
    z_in = zarr.open(in_path, mode="r")
    z_out = zarr.open(out_path, mode="r+")
    
    n_rows = z_in.shape[0]
    in_start = in_chunk_idx * chunk_rows
    in_end = min(in_start + chunk_rows, n_rows)
    
    out_start = out_chunk_idx * chunk_rows
    out_end = out_start + (in_end - in_start)
    
    # 1. Read as float64
    chunk = z_in[in_start:in_end].astype(np.float64, copy=False)

    # 2. Winsorize efficiently in-place
    # np.clip handles the upper and lower bounds simultaneously and writes back to 'chunk'
    np.clip(chunk, lower_limits, upper_limits, out=chunk)

    # 3. Rescale in-place
    # Subtract mean and divide by std directly inside the existing memory block
    with np.errstate(divide="ignore", invalid="ignore"):
        chunk -= mean
        chunk /= std

    bad_std = (std == 0.0) | np.isnan(std)
    if np.any(bad_std):
        chunk[:, bad_std] = np.where(finite_mask[:, bad_std], 0.0, np.nan)

    # 4. Write back as float16 directly to the final destination
    z_out[out_start:out_end] = chunk.astype(np.float16)


if __name__ == "__main__":
    try:
        input_path = Path(sys.argv[1])
        input_smiles_path = Path(sys.argv[2])
        outdir_path = Path(sys.argv[3])
    except:
        print("Usage: python split.py <input_zarr_path> <input_smiles_path> <outdir_path>")
        exit(1)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        exit(1)

    if not input_smiles_path.exists():
        print(f"Error: {input_smiles_path} not found.")
        exit(1)

    if outdir_path.exists():
        print(f"Warning: {outdir_path} already exists.")
        exit(1)
    else:
        outdir_path.mkdir(parents=True)

    max_workers = max(1, os.cpu_count() - 1)

    input_zarr = zarr.open(input_path, mode="r")
    input_n_chunks = input_zarr.nchunks
    chunk_indices = np.arange(input_n_chunks)[:-1]
    
    rng = np.random.default_rng(seed=42)
    rng.shuffle(chunk_indices)
    
    split_idx = int(0.9 * len(chunk_indices))
    train_chunks = chunk_indices[:split_idx]
    val_chunks = chunk_indices[split_idx:]
    rows_per_chunk = input_zarr.chunks[0]

    print("Splitting SMILES data...")
    smiles = polars.read_parquet(input_smiles_path)["SMILES"].to_list()
    train_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in train_chunks]
    val_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in val_chunks]
    polars.DataFrame({"SMILES": [s for chunk in train_smiles for s in chunk]}).write_parquet(outdir_path / "train_smiles.parquet")
    polars.DataFrame({"SMILES": [s for chunk in val_smiles for s in chunk]}).write_parquet(outdir_path / "val_smiles.parquet")

    # 1. Calculate Mean and Std straight from the source Zarr using ONLY Train Indices
    mean, std, count = mean_std_zarr_parallel(str(input_path), train_chunks, max_workers=max_workers)

    np.save(outdir_path / f"feature_train_means_{input_path.stem}.npy", mean)
    np.save(outdir_path / f"feature_train_stds_{input_path.stem}.npy", std)
    np.save(outdir_path / f"feature_train_counts_{input_path.stem}.npy", count)

    lower_limits = mean - WINSORIZATION_FACTOR * std
    upper_limits = mean + WINSORIZATION_FACTOR * std

    # 2. Create the empty destination structures
    print("Creating destination Train and Val Zarr arrays...")
    train_zarr_path = str(outdir_path / "train_rescaled.zarr")
    val_zarr_path = str(outdir_path / "val_rescaled.zarr")

    zarr.create_array(
        store=train_zarr_path,
        shape=(len(train_chunks) * rows_per_chunk, input_zarr.shape[1]),
        chunks=input_zarr.chunks,
        dtype=np.float16,
        compressors=None,
        fill_value=np.nan,
    )
    zarr.create_array(
        store=val_zarr_path,
        shape=(len(val_chunks) * rows_per_chunk, input_zarr.shape[1]),
        chunks=input_zarr.chunks,
        dtype=np.float16,
        compressors=None,
        fill_value=np.nan,
    )

    # 3. Build single-pass processing tasks
    processing_tasks = []
    
    for out_idx, in_idx in enumerate(train_chunks):
        processing_tasks.append((str(input_path), train_zarr_path, in_idx, out_idx, rows_per_chunk, lower_limits, upper_limits, mean, std))
        
    for out_idx, in_idx in enumerate(val_chunks):
        processing_tasks.append((str(input_path), val_zarr_path, in_idx, out_idx, rows_per_chunk, lower_limits, upper_limits, mean, std))

    # 4. Execute Read -> Scale -> Write in one sweep
    print(f"\nProcessing and writing scaled data in a single pass using {max_workers} processes...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_and_write_chunk, t) for t in processing_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Writing scaled chunks"):
            pass

    print("\nData splitting and scaling completed successfully.")
