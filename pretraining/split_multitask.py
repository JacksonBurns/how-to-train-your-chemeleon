import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars
import zarr
from tqdm import tqdm

from config import WINSORIZATION_FACTOR


def combine_stats(stat_a, stat_b):
    """Merge two sets of Welford statistics (n, mean, M2) into one."""
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
    """Worker function to compute stats for a single chunk."""
    zarr_path, start, end = args
    z_array = zarr.open(zarr_path, mode="r")

    chunk = z_array[start:end].astype(np.float64, copy=False)
    finite = np.isfinite(chunk)
    bcount = finite.sum(axis=0)

    if not np.any(bcount):
        n_cols = z_array.shape[1]
        return (
            np.zeros(n_cols, dtype=np.int64),
            np.zeros(n_cols, dtype=np.float64),
            np.zeros(n_cols, dtype=np.float64),
        )

    chunk_sum = np.where(finite, chunk, 0.0).sum(axis=0)
    mean = np.zeros_like(chunk_sum)
    valid = bcount > 0
    mean[valid] = chunk_sum[valid] / bcount[valid]
    diff = np.where(finite, chunk - mean, 0.0)
    m2 = (diff * diff).sum(axis=0)

    return bcount, mean, m2


def mean_std_zarr_parallel(zarr_path, train_chunks, max_workers=None):
    """Compute mean and std in parallel using ProcessPoolExecutor."""
    zarr_array = zarr.open(zarr_path, mode="r")
    n_rows, n_cols = zarr_array.shape
    chunk_rows = zarr_array.chunks[0]
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    tasks = []
    for i in range(0, n_rows, chunk_rows):
        if i // chunk_rows in train_chunks:
            end = min(i + chunk_rows, n_rows)
            tasks.append((zarr_path, i, end))

    print(f"Processing {len(tasks)} chunks with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_chunk_stats, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing chunks"):
            results.append(future.result())

    print("Merging statistics...")

    total_stats = (
        np.zeros(n_cols, dtype=np.int64),
        np.zeros(n_cols, dtype=np.float64),
        np.zeros(n_cols, dtype=np.float64),
    )
    for part_stats in results:
        total_stats = combine_stats(total_stats, part_stats)

    final_count, final_mean, final_m2 = total_stats

    variance = np.full(n_cols, np.nan, dtype=np.float64)
    valid_final = final_count > 1
    variance[valid_final] = final_m2[valid_final] / (final_count[valid_final] - 1)
    std = np.sqrt(variance)

    return final_mean, std, final_count


def rescale_zarr(
    input_zarr,
    output_path,
    chunks,
    mean,
    std,
    rows_per_chunk,
    shard_multiplier,
):
    """Apply winsorization + standardization and write rescaled Zarr."""
    lower_limits = mean - WINSORIZATION_FACTOR * std
    upper_limits = mean + WINSORIZATION_FACTOR * std

    input_rows = len(chunks) * rows_per_chunk
    bytes_per_row = input_zarr.shape[1] * 2
    target_rows_for_1gb = (1024**3) // bytes_per_row
    rows_per_shard = shard_multiplier * rows_per_chunk
    shards_shape = (rows_per_shard, input_zarr.chunks[1])

    z = zarr.create_array(
        store=output_path,
        shape=(input_rows, input_zarr.shape[1]),
        chunks=input_zarr.chunks,
        shards=shards_shape,
        zarr_format=3,
        dtype=np.float16,
        compressors=None,
        fill_value=np.nan,
    )

    for shard_start_idx in tqdm(
        range(0, len(chunks), shard_multiplier), desc=f"Rescaling {output_path.name}"
    ):
        current_shard_chunk_indices = chunks[
            shard_start_idx : shard_start_idx + shard_multiplier
        ]

        shard_data = []
        for chunk_idx in current_shard_chunk_indices:
            read_start = chunk_idx * rows_per_chunk
            read_end = read_start + rows_per_chunk

            chunk = input_zarr[read_start:read_end].astype(np.float64, copy=False)
            chunk.clip(min=lower_limits, max=upper_limits, out=chunk)
            chunk -= mean
            chunk /= std
            shard_data.append(chunk.astype(np.float16, copy=False))

        if shard_data:
            full_shard_array = np.concatenate(shard_data, axis=0)
            write_start = shard_start_idx * rows_per_chunk
            write_end = write_start + full_shard_array.shape[0]
            z[write_start:write_end] = full_shard_array


if __name__ == "__main__":
    try:
        descriptor_zarr_path = Path(sys.argv[1])
        fingerprint_zarr_path = Path(sys.argv[2])
        input_smiles_path = Path(sys.argv[3])
        outdir_path = Path(sys.argv[4])
    except:
        print(
            "Usage: python split_multitask.py "
            "<descriptor_zarr> <fingerprint_zarr> <input_smiles.parquet> <outdir>"
        )
        exit(1)

    if not descriptor_zarr_path.exists():
        print(f"Error: {descriptor_zarr_path} not found.")
        exit(1)

    if not fingerprint_zarr_path.exists():
        print(f"Error: {fingerprint_zarr_path} not found.")
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

    desc_zarr = zarr.open(descriptor_zarr_path, mode="r")
    fp_zarr = zarr.open(fingerprint_zarr_path, mode="r")

    assert desc_zarr.shape[0] == fp_zarr.shape[0], (
        f"Mismatched row counts: descriptor {desc_zarr.shape[0]} vs fingerprint {fp_zarr.shape[0]}"
    )

    input_n_chunks = desc_zarr.nchunks
    assert desc_zarr.nchunks == fp_zarr.nchunks, "Descriptor and fingerprint have different chunk counts"
    assert desc_zarr.chunks[0] == fp_zarr.chunks[0], "Descriptor and fingerprint have different row chunk sizes"

    chunk_indices = np.arange(input_n_chunks)[:-1]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(chunk_indices)
    split_idx = int(0.9 * input_n_chunks)
    train_chunks = chunk_indices[:split_idx]
    val_chunks = chunk_indices[split_idx:]
    rows_per_chunk = desc_zarr.chunks[0]

    print("Splitting data into train and validation sets...")
    smiles = polars.read_parquet(input_smiles_path)["SMILES"].to_list()
    train_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in train_chunks]
    val_smiles = [smiles[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in val_chunks]
    polars.DataFrame({"SMILES": [s for chunk in train_smiles for s in chunk]}).write_parquet(
        outdir_path / "train_smiles.parquet"
    )
    polars.DataFrame({"SMILES": [s for chunk in val_smiles for s in chunk]}).write_parquet(
        outdir_path / "val_smiles.parquet"
    )

    # Shard multiplier for I/O efficiency
    bytes_per_row = desc_zarr.shape[1] * 2
    target_rows_for_1gb = (1024**3) // bytes_per_row
    shard_multiplier = max(1, round(target_rows_for_1gb / rows_per_chunk))

    # Compute and apply rescaling for descriptors
    print("Calculating mean and std for descriptor training set...")
    desc_mean, desc_std, desc_count = mean_std_zarr_parallel(
        descriptor_zarr_path, train_chunks
    )
    np.save(outdir_path / "descriptor_train_means.npy", desc_mean)
    np.save(outdir_path / "descriptor_train_stds.npy", desc_std)
    np.save(outdir_path / "descriptor_train_counts.npy", desc_count)

    # Compute and apply rescaling for fingerprints
    print("Calculating mean and std for fingerprint training set...")
    fp_mean, fp_std, fp_count = mean_std_zarr_parallel(
        fingerprint_zarr_path, train_chunks
    )
    np.save(outdir_path / "fingerprint_train_means.npy", fp_mean)
    np.save(outdir_path / "fingerprint_train_stds.npy", fp_std)
    np.save(outdir_path / "fingerprint_train_counts.npy", fp_count)

    # Rescale both descriptor and fingerprint data
    print("Applying winsorization and rescaling to descriptor sets...")
    rescale_zarr(
        desc_zarr,
        outdir_path / "train_descriptor_rescaled.zarr",
        train_chunks,
        desc_mean,
        desc_std,
        rows_per_chunk,
        shard_multiplier,
    )
    rescale_zarr(
        desc_zarr,
        outdir_path / "val_descriptor_rescaled.zarr",
        val_chunks,
        desc_mean,
        desc_std,
        rows_per_chunk,
        shard_multiplier,
    )

    print("Applying winsorization and rescaling to fingerprint sets...")
    rescale_zarr(
        fp_zarr,
        outdir_path / "train_fingerprint_rescaled.zarr",
        train_chunks,
        fp_mean,
        fp_std,
        rows_per_chunk,
        shard_multiplier,
    )
    rescale_zarr(
        fp_zarr,
        outdir_path / "val_fingerprint_rescaled.zarr",
        val_chunks,
        fp_mean,
        fp_std,
        rows_per_chunk,
        shard_multiplier,
    )
