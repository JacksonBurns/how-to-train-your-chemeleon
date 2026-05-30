# Spec: Replace TensorBoard with MLFlow for Experiment Tracking

## Context

The pretraining pipeline currently uses `TensorBoardLogger` from PyTorch Lightning. Config values are written manually to `config.txt` in the output directory. We want to replace both with MLFlow for structured experiment tracking.

## Requirements

- Replace TensorBoard logger with MLFlow logger
- Use local file store (no external server)
- MLFlow run name must match the timestamped output directory name (`NOW` from `now.py`)
- All config parameters auto-logged as MLFlow params
- Remove the manual `config.txt` write block (redundant once MLFlow logs params)
- Add `mlflow` to the pip install command in `AGENTS.md`
- No changes to `config.py`, `now.py`, `split.py`, `dataset.py`, or feature calculators

## Changes by File

### `pretraining/train.py`

**Remove:**
- `from lightning.pytorch.loggers import TensorBoardLogger`
- Lines 202-206: `TensorBoardLogger` instantiation
- Lines 130-146: manual `config.txt` write block

**Add:**
- `import mlflow`
- `from lightning.pytorch.loggers import MLFlowLogger`
- Before creating the logger:
  ```python
  mlflow.set_tracking_uri(f"file://{output_dir / 'mlflow'}")
  ```
- Replace logger instantiation with:
  ```python
  logger = MLFlowLogger(
      experiment_name="chemeleon",
      run_name=NOW,
      tracking_uri=f"file://{output_dir / 'mlflow'}",
  )
  ```
- After creating the logger, log all config params:
  ```python
  logger.experiment.log_params(
      mlflow.active_run(),
      {
          "EPOCHS": EPOCHS,
          "FEATURIZER": FEATURIZER,
          "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
          "FNN_ACTIVATION": FNN_ACTIVATION,
          "FNN_HIDDEN_LAYERS": FNN_HIDDEN_LAYERS,
          "FNN_HIDDEN_SIZE": FNN_HIDDEN_SIZE,
          "INITIAL_LEARNING_RATE": INITIAL_LEARNING_RATE,
          "MAXIMUM_LEARNING_RATE": MAXIMUM_LEARNING_RATE,
          "MP_ACTIVATION": MP_ACTIVATION,
          "MP_DEPTH": MP_DEPTH,
          "MP_HIDDEN_SIZE": MP_HIDDEN_SIZE,
          "PATIENCE": PATIENCE,
          "WARMUP_EPOCHS": WARMUP_EPOCHS,
          "CHUNKS_PER_BATCH": CHUNKS_PER_BATCH,
          "DROPOUT_FRACTION": DROPOUT_FRACTION,
          "WINSORIZATION_FACTOR": WINSORIZATION_FACTOR,
      },
  )
  ```
- Pass `logger=logger` to `Trainer(...)` instead of `logger=tensorboard_logger`

**Import updates needed:**
- Add `DROPOUT_FRACTION` and `WINSORIZATION_FACTOR` to the `from config import (...)` block

### `AGENTS.md`

Update the pip install line to include `mlflow`:
```
pip install 'chemprop>=2.2.3' zarr polars pyarrow tensorboard cuik-molmaker mlflow --extra-index-url https://pypi.nvidia.com/rdkit-latest/
```

## Verification

After implementation:
1. Run `python pretraining/train.py <input_dir> <output_dir>` on a small test dataset
2. Confirm `<output_dir>/<timestamp>/mlflow/` directory exists with run data
3. Confirm `mlflow ui --backend-store-uri <output_dir>/<timestamp>/mlflow/` surfaces the run with all params logged
4. Confirm `config.txt` is no longer written to the output directory

## Notes

- `mlflow.set_tracking_uri()` must be called before `MLFlowLogger` instantiates, otherwise the logger defaults to `./mlruns`
- `DROPOUT_FRACTION` and `WINSORIZATION_FACTOR` are currently not imported in `train.py` — they need to be added to the import statement
- `tensorboard` can remain in the pip install line (no harm keeping it), or can be dropped — either is acceptable
