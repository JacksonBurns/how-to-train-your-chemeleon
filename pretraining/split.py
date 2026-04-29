import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    This allows parallel computation of variance.
    """
    na, mean_a, m2_a = stat_a
    nb, mean_b, m2_b = stat_b

    n_combined = na + nb

    # Calculate delta
    delta = mean_b - mean_a

    # Combined mean
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_combined = mean_a + delta * (nb / n_combined)

        # Combined M2 (Sum of squares of differences from the mean)
        m2_combined = m2_a + m2_b + (delta**2) * (na * nb / n_combined)

    # Where only one side had data, we preserve that side's stats
    mask_a_only = (na > 0) & (nb == 0)
    mask_b_only = (nb > 0) & (na == 0)

    # Restore values where the other side was empty (fix NaNs from 0/0)
    mean_combined[mask_a_only] = mean_a[mask_a_only]
    m2_combined[mask_a_only] = m2_a[mask_a_only]

    mean_combined[mask_b_only] = mean_b[mask_b_only]
    m2_combined[mask_b_only] = m2_b[mask_b_only]

    # Fill remaining NaNs (where both were 0) with 0.0
    mean_combined = np.nan_to_num(mean_combined, nan=0.0)
    m2_combined = np.nan_to_num(m2_combined, nan=0.0)

    return n_combined, mean_combined, m2_combined


def compute_chunk_stats(args):
    """
    Worker function to compute stats for a single chunk in float64.
    """
    zarr_path, start, end = args

    z_array = zarr.open(zarr_path, mode="r")

    # Read chunk and explicitly upcast to float64 to prevent overflow
    chunk = z_array[start:end].astype(np.float64, copy=False)

    # Mask finite values
    finite = np.isfinite(chunk)
    bcount = finite.sum(axis=0)

    n_cols = z_array.shape[1]
    
    # If chunk has no valid data at all
    if not np.any(bcount):
        return (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))

    # Compute local mean
    chunk_sum = np.where(finite, chunk, 0.0).sum(axis=0)

    mean = np.zeros_like(chunk_sum, dtype=np.float64)
    valid = bcount > 0
    mean[valid] = chunk_sum[valid] / bcount[valid]

    # Compute local M2
    diff = np.where(finite, chunk - mean, 0.0)
    m2 = (diff * diff).sum(axis=0)

    return bcount, mean, m2


def mean_std_zarr_parallel(zarr_path, max_workers=None):
    """
    Computes mean and std in parallel using ProcessPoolExecutor.
    Maintains float64 precision throughout calculation.
    """
    zarr_array = zarr.open(zarr_path, mode="r")
    n_rows, n_cols = zarr_array.shape
    chunk_rows = zarr_array.chunks[0]

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    # Prepare tasks
    tasks = []
    for i in range(0, n_rows, chunk_rows):
        end = min(i + chunk_rows, n_rows)
        tasks.append((zarr_path, i, end))

    print(f"Processing {len(tasks)} chunks with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_chunk_stats, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing chunks"):
            results.append(future.result())

    print("Merging statistics...")

    # Initialize float64 accumulator
    total_stats = (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))

    for part_stats in results:
        total_stats = combine_stats(total_stats, part_stats)

    final_count, final_mean, final_m2 = total_stats

    # Calculate final Std Dev
    variance = np.full(n_cols, np.nan, dtype=np.float64)
    valid_final = final_count > 1

    variance[valid_final] = final_m2[valid_final] / (final_count[valid_final] - 1)
    std = np.sqrt(variance)

    return final_mean, std, final_count


def copy_chunk(args):
    """ Worker function to copy a chunk from input to output Zarr in parallel. """
    in_path, out_path, in_start, out_start, rows_per_chunk = args
    z_in = zarr.open(in_path, mode="r")
    z_out = zarr.open(out_path, mode="r+")
    
    in_end = in_start + rows_per_chunk
    out_end = out_start + rows_per_chunk
    
    z_out[out_start:out_end, :] = z_in[in_start:in_end, :]


def rescale_chunk(args):
    """ Worker function to winsorize and rescale chunks in parallel. """
    zarr_path, start, end, lower_limits, upper_limits, mean, std = args
    z = zarr.open(zarr_path, mode="r+")
    
    # Read as float64 to prevent overflow
    chunk = z[start:end].astype(np.float64, copy=False)
    finite_mask = np.isfinite(chunk)

    # Winsorize: Apply clipping only to finite values
    chunk = np.where((chunk < lower_limits) & finite_mask, lower_limits, chunk)
    chunk = np.where((chunk > upper_limits) & finite_mask, upper_limits, chunk)

    # Rescale
    with np.errstate(divide="ignore", invalid="ignore"):
        rescaled = (chunk - mean) / std

    # Restore NaNs where original data was completely absent
    rescaled = np.where(finite_mask, rescaled, np.nan)

    # Handle zero-variance or completely absent columns
    # If std is 0.0 or NaN, standardize finite entries in those columns to 0.0 
    bad_std = (std == 0.0) | np.isnan(std)
    if np.any(bad_std):
        rescaled[:, bad_std] = np.where(finite_mask[:, bad_std], 0.0, np.nan)

    # Write back as float16
    z[start:end] = rescaled.astype(np.float16)


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
        print(f"Warning: {outdir_path} already exists. Overwrite contents? (y/n)")
        response = input().strip().lower()
        if response != "y":
            print("Operation cancelled.")
            exit(1)
    else:
        outdir_path.mkdir(parents=True)

    # Set up globals for processes
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

    print("Creating Train and Validation Zarr structures...")
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

    # Prepare copy tasks
    copy_tasks = []
    for out_idx, in_idx in enumerate(train_chunks):
        in_start = in_idx * rows_per_chunk
        out_start = out_idx * rows_per_chunk
        copy_tasks.append((str(input_path), train_zarr_path, in_start, out_start, rows_per_chunk))
        
    for out_idx, in_idx in enumerate(val_chunks):
        in_start = in_idx * rows_per_chunk
        out_start = out_idx * rows_per_chunk
        copy_tasks.append((str(input_path), val_zarr_path, in_start, out_start, rows_per_chunk))

    print(f"Copying data into distinct Train and Val sets using {max_workers} processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_chunk, t) for t in copy_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying chunks"):
            pass

    print("\nCalculating mean and std for training set (Float64 Precision)...")
    mean, std, count = mean_std_zarr_parallel(train_zarr_path, max_workers=max_workers)

    # Save metadata (will be float64 type)
    np.save(outdir_path / f"feature_train_means_{input_path.stem}.npy", mean)
    np.save(outdir_path / f"feature_train_stds_{input_path.stem}.npy", std)
    np.save(outdir_path / f"feature_train_counts_{input_path.stem}.npy", count)

    lower_limits = mean - WINSORIZATION_FACTOR * std
    upper_limits = mean + WINSORIZATION_FACTOR * std

    # Prepare parallel rescale tasks
    rescale_tasks = []
    for zarr_path in [train_zarr_path, val_zarr_path]:
        z = zarr.open(zarr_path, mode="r")
        n_rows = z.shape[0]
        chunk_rows = z.chunks[0]
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            rescale_tasks.append((zarr_path, start, end, lower_limits, upper_limits, mean, std))

    print("\nApplying winsorization and rescaling to train and validation sets in parallel...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(rescale_chunk, t) for t in rescale_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Rescaling chunks"):
            pass

    print("\nData splitting and scaling completed successfully.")
