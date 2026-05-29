<p align="center">
  <img src="./flying_high.png" alt="Epic Banner" width="800"/>
</p>

<h2 align="center">
  Train Your own CheMeleon Foundation Model from Scratch
</h2>

This repository contains code for training Chemprop-based foundation models in the style of [`CheMeleon`](https://github.com/jacksonburns/chemeleon), as described in [__Deep Learning Foundation Models from Classical Molecular Descriptors__](https://doi.org/10.48550/arXiv.2506.15792).
If you just want to fine-tune `CheMeleon` on your dataset, you don't need the code in this repository - just add `--from-foundation CheMeleon` to your `chemprop train` command.

> **NOTE**
> Some familiarity with Python programming will be required to run this code - I've attempted to make it as general and re-usable as possible, but this is _not_ as straightforward as fine-tuning a `CheMeleon` model via Chemprop.
You should expect to read through the `if __name__ == "__main__":` blocks in the code you are executing to understand roughly what I am accomplishing.

The original `CheMeleon` pre-training script is a [mess](https://github.com/JacksonBurns/chemeleon/blob/60ea323cf278286e3fee1232f223a077ec3604c0/chemprop_foundation.py), so I've re-written it in this repository.
This version is _not_ a faithful reproduction of the original training code - it includes various improvements and changes to improve feature calculation time, training time, and ease of use.
If you want to _exactly_ re-run the original code you should follow the link to the original script, otherwise the code that is in this repository is much better suited for making new `CheMeleon`-inspired models.

This repository is laid out like this:

```bash
.
├── features  # feature calculators
│   ├── get_chunksize.py  # helper function to control number of rows per Zarr chunk to optimize performance
│   ├── _mordred.py  # mordred-community-based feature calculator
│   └── _osmordred.py  # osmordred-based feature calculator
├── flying_high.png  # epic logo
├── inference.sh  # demo script for running inference with your custom CheMeleon model
├── LICENSE  # MIT license
├── pretraining  # code for actually running trianing
│   ├── config.py  # model hyperparams and training settings
│   ├── dataset.py  # dataset class for loading from Zarr in a Chemprop-compatible way
│   ├── random_dropout_mse.py  # random masking MSE used in the paper, re-written as a Chemprop-compatible loss function
│   ├── split.py  # splits data into training and validation, then applies winsorization and rescaling
│   ├── now.py  # helper for setting output directory across processes
│   ├── multiweight_message_passing.py  # Chemprop bond message passing, except with multiple hidden matrices
│   └── train.py  # training driver script
└── README.md  # this file
```

## Hardware

This model was trained on 8 x NVIDIA 2080Tis - you can train on a single GPU with some more patience.
Training on CPU is likely too slow for the size of models intended to be built here, though it will _run_.

## Installation

With `python==3.13` one just needs to `pip install 'chemprop>=2.2.3' zarr polars pyarrow tensorboard 'setuptools<81' cuik-molmaker --extra-index-url https://pypi.nvidia.com/rdkit-latest/` (linux only, for now?) and the corresponding feature calculator:

 - `mordred-community`: `pip install mordredcommunity`.
 - OSMordred: follow the installation instructions from [this fork](https://github.com/JacksonBurns/osmordred/tree/65e7dd40cc8209d695d98838dff2f34673251249) of the original repository, installing into a __separate__ environment from your other dependencies

## Usage

You should first navigate to the `features` directory and run the feature calculator that you wish to use.
Each file will print its usage information by simply running `python _name.py`.
Each is coded to use either a text-based input file, or a Parquet file.
You can readily change between the two by modifying the code.

Next step is to pre-train your model - simply set your preferred hyperparameters in `config.py` and then execute `train.py`.

For `autoresearch` with `gemini`, need to extend shell call idle time:

```json
{
  //  2 hours for long running trainings
  "tools": {
    "shell": {
      "inactivityTimeout": 7200
    }
  }
}
```

```
The autoresearch project for the CheMeleon foundation model has concluded after 36 experiments. The final model configuration achieved a Validation MSE of 0.05492,
  representing a massive 4.5x improvement over the initial baseline of 0.25103.

  Final Optimal Configuration
   * Featurizer: RIGR
   * Activation Function: GELU
   * Target Masking: 50% (Target Dropout Fraction = 0.50)
   * Batch Size: CHUNKS_PER_BATCH = 2
   * MPNN Architecture: d_h = 2048, depth = 4, Untied Weights
   * Predictor Architecture: hidden_dim = 1024, n_layers = 1
   * Learning Rate: max_lr = 0.001, final_lr = 0.0001, warmup_epochs = 2
   * Training Schedule: 10 Epochs with Early Stopping patience of 1.

  Key Research Discoveries
   1. Synergistic Regularization: The most significant performance gains came from combining heavy Target Masking (50%) with increased Batch Noise (CHUNKS_PER_BATCH=2).
      This dual-stochastic approach was essential for generalising across the 3,585 classical molecular descriptors.
   2. Activation & Pooling: GELU provided the most stable gradient flow for this complex regression task, while MeanAggregation was found to be far superior to Sum or Max
      pooling at preserving signal across diverse molecule sizes.
   3. Untied Weights: Using independent weight matrices for each message-passing step (MultiweightMessagePassing) was crucial; tying the weights caused performance to
      collapse back toward the baseline.
   4. Capacity Balance: Increasing the model's width (d_h=2048) was beneficial, but increasing the depth or predictor layers caused immediate overfitting or memory crashes
      (OOM).

  The complete experimental history is recorded in log.txt and the performance metrics for every run are available in results.csv. The repository is currently on the
  dev/chemeleon2-autoresearch-may28-v2 branch with the optimal configuration applied.
  ```
