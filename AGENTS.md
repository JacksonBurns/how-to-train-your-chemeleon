# AGENTS.md

## Quick Overview

Python 3.13 research repo for training CheMeleon-style molecular foundation models. No package manager, no tests, no CI. Two phases: compute molecular descriptors (features/), then train (pretraining/).

## Setup

- No `requirements.txt` or `pyproject.toml`. Install manually:
  ```
  pip install 'chemprop>=2.2.3' zarr polars pyarrow tensorboard cuik-molmaker mlflow --extra-index-url https://pypi.nvidia.com/rdkit-latest/
  ```
- Feature calculator: `pip install mordredcommunity` (for _mordred.py) or OSMordred in a __separate__ environment (for _osmordred.py).
- Linux only. GPU required (trained on 8x 2080Ti). CPU is impractical.

## Two-Phase Workflow

### Phase 1: Feature Calculation

```
python features/_mordred.py <smiles_file> <output_zarr.zarr>
```
Outputs a Zarr array of molecular descriptors. Each calculator prints its own usage via `python <file>`. Input can be text or Parquet — check `__main__` blocks to switch.

### Phase 2: Data Split + Rescaling

```
python pretraining/split.py <input_zarr.zarr> <smiles.parquet> <output_dir>
```
Splits data 90/10 by Zarr chunks, computes mean/std on training set, applies winsorization (factor=3) and rescaling. Outputs `train_rescaled.zarr`, `val_rescaled.zarr`, `train_smiles.parquet`, `val_smiles.parquet`. Refuses to overwrite existing output dir.

### Phase 3: Training

```
cd pretraining
python train.py <input_dir> <output_dir>
```
`input_dir` must contain `train_rescaled.zarr`, `val_rescaled.zarr`, `train_smiles.parquet`, `val_smiles.parquet`. Output is timestamped subdirectory (from `now.py`), or use `CHEMELEON_SHARED_OUTPUT_DIR` env var. Saves `best.pt` and `chemeleon2_preview_mp.pt`.

## Architecture

- **`config.py`** — all hyperparameters as module-level constants. No CLI args. Edit this file to change training settings.
- **`dataset.py`** — `ChempropChunkwiseZarrDataset` reads Zarr in multi-chunk batches. `zarr.open_array` is lazy-loaded per worker to avoid multiprocessing hangs.
- **`split.py`** — parallel Welford online statistics across Zarr chunks. Uses `ProcessPoolExecutor`, passes file paths (not objects) to workers.
- **`train.py`** — PyTorch Lightning trainer. `PatchedCuikmolmakerMolGraphFeaturizer` injects dummy nodes for empty SMILES. Custom `MultiweightMessagePassing` unties the hidden weight matrix per depth step. Custom `RandomDropoutMSE` loss masks 70% of features randomly each batch.
- **`now.py`** — generates timestamped output dir names. Shared across processes via `CHEMELEON_SHARED_OUTPUT_DIR` env var.

## Key Conventions

- All scripts are entry points, run as `python <file>.py`. No imports between phases.
- All scripts use `sys.argv` for I/O paths, with a bare `except:` fallback that prints usage.
- Zarr format v3, no compression, float16 for rescaled data, float32 for raw features.
- SMILES column in Parquet must be named `"SMILES"`.
- `PatchedCuikmolmakerMolGraphFeaturizer` requires `FEATURIZER` from config (V2 or RIGR).
