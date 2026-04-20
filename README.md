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
<!-- 
TODO: update the below !!

 - `random_dropout_mse.py`: the custom loss function used in `CheMeleon`, which randomly drops out a user-configurable fraction of the targets during training as a form of regularization.
 This one is slightly different from that in the original paper, as it allows us to _ignore_ missing features rather than imputing them, as was done in the original paper.
 - `features`: feature calculators used to generate the pre-training targets, as well as the corresponding Chemprop-compatible loaders for actually running training on those features.
 This repository includes the [OSmordred](https://github.com/osmoai/osmordred) feature calculator as well as the original [`mordred-community`](https://github.com/JacksonBurns/mordred-community) feature calculator from the `CheMeleon` paper.

> **NOTE**
> ~~There are two versions of the loader: a 'normal' version used in the original paper and a 'chunked' version.
The latter loads chunks out of the serialized data array (in Zarr format) at a time, rather than accessing many random rows.
This makes training roughly ~100x faster _but_ means that training is no longer _truly_ stochastic, and instead pseudo-random.
This is included to help with scale - whether or not this approximation introduces too much error has not been verified.~~ <-- update this I removed the old one

 - `config.py`: exposes all of the hyperparameters that one might reasonably consider changing to effect the performance of `CheMeleon`.
 The original paper did very minimal tuning of these!
 - `train.py`: driver code to actually execute the training. -->

## Hardware



## Installation

With `python==3.13` one just needs to `pip install 'chemprop>=2.2.3' zarr polars` and the corresponding feature calculator:

 - `mordred-community`: `pip install mordred-community`.
 - OSMordred: follow the installation instructions from [this fork](https://github.com/JacksonBurns/osmordred/tree/65e7dd40cc8209d695d98838dff2f34673251249) of the original repository, installing into a __separate__ environment from your other dependencies

## Usage

You should first navigate to the `features` directory and run the feature calculator that you wish to use.
Each file will print its usage information by simply running `python _name.py`.
Each is coded to use either a text-based input file, or a Parquet file.
You can readily change between the two by modifying the code.

Next step is to pre-train your model - simply set your preferred hyperparameters in `config.py` and then execute `train.py`.

Finally, running fine-tuning using using your pre-trained model.
If you use `FEATURIZER = "default"`, you can use your trained model in the Chemprop CLI with just the `--from-foundation` argument.
If you switch to `FEATURIZER = "rigr"`, you will need to add that option in the CLI with `--multi-hot-atom-featurizer-mode RIGR`.
