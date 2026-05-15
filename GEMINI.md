# Project Overview

This repository provides an improved and re-written framework for training Chemprop-based foundation models, inspired by the original `CheMeleon` project. Its primary goal is to facilitate the pre-training of models designed to learn molecular descriptors, crucial for various cheminformatics tasks. This version offers enhancements over the original script, focusing on improved feature calculation, training efficiency, and overall ease of use.

The project is primarily developed in Python and leverages several key libraries from the scientific computing and machine learning ecosystem.

## Key Technologies

*   **Python:** The core language for the entire project.
*   **Chemprop:** A powerful library for molecular property prediction using graph neural networks, serving as the foundation for the models.
*   **PyTorch Lightning:** Used in `pretraining/train.py` for efficient and organized model training.
*   **Zarr:** Employed for efficient storage and access of large numerical arrays, particularly for molecular features.
*   **Polars:** A fast DataFrame library used for data manipulation, especially with SMILES files.
*   **Mordred-community / OSMordred:** Libraries for calculating a wide range of molecular descriptors, used in the `features` directory.
*   **RDKit:** Essential for cheminformatics tasks, molecular manipulation, and SMILES processing.
*   **cuik-molmaker:** A custom library for molecular graph featurization, integrated into the training pipeline.

## Architecture

The project is logically divided into two main components:

### `features` directory
This directory contains scripts responsible for the computation of molecular descriptors. These scripts take SMILES strings as input, calculate the specified features (e.g., using Mordred), and store them in Zarr format for efficient access during training.

*   `_mordred.py`: A script for streaming Mordred descriptor computation.
*   `_osmordred.py`: An alternative script for OSMordred-based feature calculation.
*   `get_chunksize.py`: A helper function to optimize Zarr chunk sizes for performance.

### `pretraining` directory
This section houses the core logic for model training, data handling, and configuration.

*   `config.py`: Defines global hyperparameters for data preparation, training cycles, learning rates, and the architecture of the neural network (Feed-Forward Network and Message Passing Neural Network).
*   `dataset.py`: Manages the loading of data from Zarr stores in a format compatible with Chemprop models.
*   `train.py`: The main training driver script. It orchestrates the entire training process, including dataset loading, model initialization (using `MPNN` with custom components like `MultiweightMessagePassing` and `RegressionFFN`), loss function definition (`RandomDropoutMSE`), and utilizes PyTorch Lightning's `Trainer` for execution, logging, and checkpointing.
*   `random_dropout_mse.py`: Implements a custom random masking Mean Squared Error loss function.
*   `multiweight_message_passing.py`: A specialized Chemprop bond message passing implementation with multiple hidden matrices.
*   `split.py`: Handles data splitting into training and validation sets, along with winsorization and rescaling.

## Building and Running

### Hardware Considerations
The original models were trained on 8 x NVIDIA 2080Tis. Training on a single GPU is feasible but will require more patience. CPU-only training is generally too slow for the intended model sizes.

### Installation

It is recommended to use `python==3.12`. The core dependencies can be installed using `pip`:

```bash
pip install 'chemprop>=2.2.3' zarr polars pyarrow tensorboard 'setuptools<81' cuik-molmaker --extra-index-url https://pypi.nvidia.com/rdkit-latest/
```

Additionally, you need to install the specific feature calculator you intend to use:

*   **mordred-community:**
    ```bash
    pip install mordredcommunity
    ```
*   **OSMordred:** Follow the installation instructions provided in [this specific fork](https://github.com/JacksonBurns/osmordred/tree/65e7dd40cc8209d695d98838dff2f34673251249) of the original repository. **Install this into a separate environment** to avoid dependency conflicts.

### Usage

The typical workflow involves two main steps:

1.  **Feature Calculation:**
    Navigate to the `features` directory and run the desired feature calculator script. Each script provides usage information when run without arguments (e.g., `python _mordred.py`).
    ```bash
    python features/_mordred.py <SMILES_FILE> <OUTPUT_ZARR_PATH>
    ```
    or
    ```bash
    python features/_osmordred.py <SMILES_FILE> <OUTPUT_ZARR_PATH>
    ```

2.  **Model Pre-training:**
    Before training, you should configure the model hyperparameters and training settings in `pretraining/config.py` to match your requirements.
    Then, execute the main training driver script:
    ```bash
    python pretraining/train.py <input_data_directory> <output_directory_for_model_and_logs>
    ```
    The `<input_data_directory>` should contain the `train_rescaled.zarr`, `val_rescaled.zarr`, `train_smiles.parquet`, and `val_smiles.parquet` files generated from your feature calculation and data splitting steps. The `<output_directory_for_model_and_logs>` is where the trained model checkpoints and TensorBoard logs will be saved.

## Development Conventions

*   **Logging:** `lightning.pytorch.loggers.TensorBoardLogger` is used for logging training metrics.
*   **Early Stopping:** Training utilizes `EarlyStopping` based on validation MSE to prevent overfitting.
*   **Model Checkpointing:** `ModelCheckpoint` saves the best performing models during training.
*   **Python Best Practices:** The code generally follows Python best practices, including type hinting and modular design.
*   **JIT Compilation:** The model is compiled using `torch.compile` for potential performance improvements.
