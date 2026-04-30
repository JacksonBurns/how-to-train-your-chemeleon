import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Worker function to compute stats for a single chunk.
    Args:
        args: tuple of (zarr_path, start_row, end_row)
    """
    zarr_path, start, end = args
    z_array = zarr.open(zarr_path, mode="r")
    
    # Upcast to float64 to maintain precision during statistical calculations
    chunk = z_array[start:end].astype(np.float64, copy=False)
    finite = np.isfinite(chunk)
    bcount = finite.sum(axis=0)
    
    if not np.any(bcount):
        n_cols = z_array.shape[1]
        return (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))
        
    chunk_sum = np.where(finite, chunk, 0.0).sum(axis=0)
    mean = np.zeros_like(chunk_sum)
    valid = bcount > 0
    mean[valid] = chunk_sum[valid] / bcount[valid]
    diff = np.where(finite, chunk - mean, 0.0)
    m2 = (diff * diff).sum(axis=0)
    
    return bcount, mean, m2


def mean_std_zarr_parallel(zarr_path, train_chunks, max_workers=None):
    """
    Computes mean and std in parallel using ProcessPoolExecutor.
    Computations are explicitly performed in float64.
    """
    zarr_array = zarr.open(zarr_path, mode="r")
    n_rows, n_cols = zarr_array.shape
    chunk_rows = zarr_array.chunks[0]
    if max_workers is None:
        # Leave a couple of cores free for the system/coordinator
        max_workers = max(1, os.cpu_count() - 1)

    # Prepare tasks
    tasks = []
    for i in range(0, n_rows, chunk_rows):
        if i // chunk_rows in train_chunks:
            end = min(i + chunk_rows, n_rows)
            # We pass the path, not the object, to avoid pickling large Zarr objects
            tasks.append((zarr_path, i, end))

    print(f"Processing {len(tasks)} chunks with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(compute_chunk_stats, t) for t in tasks]

        # Gather results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing chunks"):
            results.append(future.result())

    print("Merging statistics...")

    # Reduce step: Combine all partial stats
    # Initialize accumulator with zeros using float64
    total_stats = (np.zeros(n_cols, dtype=np.int64), np.zeros(n_cols, dtype=np.float64), np.zeros(n_cols, dtype=np.float64))

    # Iteratively merge
    for part_stats in results:
        total_stats = combine_stats(total_stats, part_stats)

    final_count, final_mean, final_m2 = total_stats

    # Calculate final Std Dev
    variance = np.full(n_cols, np.nan, dtype=np.float64)
    valid_final = final_count > 1

    # Variance = M2 / (n - 1) for sample variance
    variance[valid_final] = final_m2[valid_final] / (final_count[valid_final] - 1)
    std = np.sqrt(variance)

    # Returning as float64 natively
    return final_mean, std, final_count


if __name__ == "__main__":
    import sys
    from pathlib import Path

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
        print("Operation cancelled.")
        exit(1)
    else:
        outdir_path.mkdir(parents=True)

    input_zarr = zarr.open(input_path, mode="r")
    train_zarr = outdir_path / "train_rescaled.zarr"
    val_zarr = outdir_path / "val_rescaled.zarr"
    input_n_chunks = input_zarr.nchunks
    # randomly choose 90% of chunks for training, 10% for validation, skipping the last (potentially partial) chunk for simplicity
    chunk_indices = np.arange(input_n_chunks)[:-1]
    rng = np.random.default_rng(seed=42)  # for reproducibility
    rng.shuffle(chunk_indices)
    split_idx = int(0.9 * input_n_chunks)
    train_chunks = chunk_indices[:split_idx]
    val_chunks = chunk_indices[split_idx:]
    rows_per_chunk = input_zarr.chunks[0]

    # load smiles, split by chunk, and save to new files for train and val sets
    print("Splitting data into train and validation sets...")
    smiles = polars.read_parquet(input_smiles_path)["SMILES"].to_list()
    train_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in train_chunks]
    val_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in val_chunks]
    polars.DataFrame({"SMILES": [s for chunk in train_smiles for s in chunk]}).write_parquet(outdir_path / "train_smiles.parquet")
    polars.DataFrame({"SMILES": [s for chunk in val_smiles for s in chunk]}).write_parquet(outdir_path / "val_smiles.parquet")

    print("Calculating mean and std for training set...")
    mean, std, count = mean_std_zarr_parallel(input_path, train_chunks)

    # save metadata (will naturally save as float64 now)
    np.save(outdir_path / f"feature_train_means_{input_path.stem}.npy", mean)
    np.save(outdir_path / f"feature_train_stds_{input_path.stem}.npy", std)
    np.save(outdir_path / f"feature_train_counts_{input_path.stem}.npy", count)

    # calculate winsorization limits
    lower_limits = mean - WINSORIZATION_FACTOR * std
    upper_limits = mean + WINSORIZATION_FACTOR * std
    
    # shard logic
    bytes_per_row = input_zarr.shape[1] * 2
    target_rows_for_1gb = (1024**3) // bytes_per_row
    shard_multiplier = max(1, round(target_rows_for_1gb / rows_per_chunk))
    rows_per_shard = shard_multiplier * rows_per_chunk
    shards_shape = (rows_per_shard, input_zarr.chunks[1])

    # apply rescaling and winsorization
    print("Applying winsorization and rescaling to train and validation sets...")
    for zarr_chunks, zarr_path in zip([train_chunks, val_chunks], [train_zarr, val_zarr]):
        z = zarr.create_array(
            store=zarr_path,
            shape=(len(zarr_chunks) * rows_per_chunk, input_zarr.shape[1]),
            chunks=input_zarr.chunks,
            shards=shards_shape,
            zarr_format=3,
            dtype=np.float16,
            compressors=None,
            fill_value=np.nan,
        )
        n_rows = z.shape[0]
        chunk_rows = z.chunks[0]
        for start in tqdm(range(0, n_rows, chunk_rows), desc=f"Rescaling {zarr_path.name}"):
            end = min(start + chunk_rows, n_rows)
            
            # Upcast block to float64 to prevent bounds errors / type mismatch during operations with float64 means/stds
            chunk = input_zarr[start:end].astype(np.float64, copy=False)
            
            # Winsorize
            chunk.clip(min=lower_limits, max=upper_limits, out=chunk)
            
            # Rescale
            chunk -= mean
            chunk /= std
            
            # Cast down to float16 purely for saving to the Zarr store
            z[start:end] = chunk.astype(np.float16, copy=False)
