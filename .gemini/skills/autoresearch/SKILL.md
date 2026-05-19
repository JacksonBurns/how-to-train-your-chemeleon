---
name: autoresearch
description:
   Autonomously research how to improve CheMeleon
---

# autoresearch

You are an AI assistant whose job is to autonomously perform research (an "experiment") towards improving the `CheMeleon` foundation model, eventually arriving at `CheMeleon2`.

> **NOTE**
> This research involves running many long-running shell commands - DO NOT run these commands in the background, just run them in the foreground and wait patiently for them to finish before checking results, reading output files, etc.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`).
2. **Create the branch**: checkout a new branch with a name derived from the current branch, i.e. `git checkout -b <current_name>-autoresearch-mar5`
3. **Get the data**: ask the user for the location of the pre-split training and validation data from the `split.py` script
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` and `GEMINI.md` — repository context.
   - `pretraining/` - the files you modify.
   Model architecture, optimizer, training loop, etc. are all defined in here.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on this 8-GPU machine named `mithrim` and is launched from the script `train.py` within the `pretraining` directory.
This script requires 2 inputs: the pre-split training and validation data (you should ask the user for this) and the output directory (which should just be set to `chemeleon2-autoresearch-mar5` or i.e. match the branch name)

**What you CAN do:**

 - Modify model architecture by changing `pretraining/config.py`, `pretraining/multiweight_message_passing.py`, `pretraining/random_dropout_mse.py`, and `pretraining/train.py`.
 Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

> **NOTE**
> You are optimizing the architecture on a subset of the available training data - the true scale is about 50x larger, so some parameters like batch size may not be as pertinent.

**What you CANNOT do:**

 - Modify the evaluation code (`evaluate.py`) **EXCEPT** to make it compatible with the model loading logic, e.g. aggregation function, according to your changes
 - Modify the utility code in `now.py` and `split.py`.
 These are not related to the model or training, and should be left alone.
 - Modify training or validation data - this should remain as exactly what the user requests.
 - Install new packages or add dependencies. You can only use what's already installed in the `httyc` conda environment.

**The goal is simple: get the best evaluation performance.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size.
The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful performance gains, but it should not blow up dramatically.
Arbitrarily large increases may cause the run to crash.
If a run crashes due to OOM, you can try to fix it by reducing batch size or model size, but if the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.
Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.
When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.
A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.
An improvement of ~0 but much simpler code? Keep.

**Exploration diversity**: Don't just tweak hyperparameters given in `config.py`.
Alternate between categories: architecture (depth, width, attention), optimization (LR, batch size, schedules), regularization, and simplification (removing components).
If your last 3 experiments were all in the same category, try a different one.
Edit the training routine and model architecture in ways not reflected in the `config.py` file - anything to improve the model.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Experiment Commands and Output Format

All of these commands should be run from the `pretraining` directory within this repository.
Many will redirect all output to `train_output.log` (or similar) which you can read after the run finishes.
Note that each of these commands can take a *long time*.
You should run these in the foreground and wait for the output - DO NOT send them to background, just wait patiently for them to exit and then check status, read output files, etc.

Your first step in each experiment, running the training script, should look like this:

```bash
conda run --no-capture-output -n httyc python train.py /path/to/data_split chemeleon2 &> train_output.log
```

The last two lines of the file (read them with `tail`) tell you where to find the best model checkpoint (the one you will use for evaluation) and what the validation performance was for that checkpoint.
Example:

```bash
Best model validation metrics: [{'val/mse': 0.2836931347846985, 'val/mae': 0.26995477080345154, 'val/r2': 0.30225974321365356, 'val/rmse': 0.5326285362243652, 'val_loss': 0.3268914520740509}]
Best model file: /home/jwburns/how-to-train-your-chemeleon/pretraining/chemeleon2/2026-05-18_18-01-04/checkpoints/epoch=0-step=182.ckpt
```

You should then run the `extract_mp.py` file to retrieve the message passing weights (the actual foundation model).
This script will also print the filepath to the new weights.
Example:

```bash
$conda run --no-capture-output -n httyc python extract_mp.py chemeleon2/2026-05-15_15-20-51/checkpoints/epoch\=0-step\=750.ckpt 

output will be written to '/home/jwburns/how-to-train-your-chemeleon/pretraining/chemeleon2/2026-05-15_15-20-51/checkpoints/epoch=0-step=750_mp.pt'
```

Finally, use this command (notice the different conda environment and the `CUDA_VISIBLE_DEVICES=3` prefix) to run the evaluation script on the extracted message passing weights.

```bash
$ CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n polaris python evaluate.py chemeleon2/2026-05-15_15-20-51/checkpoints/epoch\=0-step\=750_mp.pt &> eval_output.log
```

The evaluation script writes a file called `results.txt` which contains the evaluation metrics for this experimentin particular:

```
SUMMARY_METRIC: 0.997900
```

**Higher is better for this metric**.
You should extract this summary metric and record it in the `results.tsv` file along with the pretraining MSE and a short description of what you changed in this experiment.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and these columns:

```
commit	performance  pretraining mse	status	description
```

1. git commit hash (short, 7 chars)
2. evaluation performance achieved (e.g. 0.750) — use a blank for crashes
3. pretraining MSE (the `val/mse` metric from the training log) — use a blank for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	performance  pretraining mse	status	description
a1b2c3d	0.997900	0.283693	keep	baseline
b2c3d4e	0.993200	0.326891	keep	increase LR to 0.04
c3d4e5f	0.990000	0.345551	discard	switch to GeLU activation
d4e5f6g			crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `pretraining/config.py`, `pretraining/multiweight_message_passing.py`, `pretraining/random_dropout_mse.py`, and `pretraining/train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment by following the instructions in the [Experiment Commands and Output Format](#experiment-commands-and-output-format) section above, making sure to use the redirect output to a log file so that it doesn't flood your context.
5. Read out the results from `results.txt` which will be written by `evaluate.py` on each run.
6. If the run crashed at any stage, read the Python stack trace and attempt a fix.
If you can't get things to work after more than two attempts, give up.
7. Record the results in the tsv (NOTE: commit the results.tsv file to git can track it)
8. If performance improved, you "advance" the branch, keeping the git commit
9. If performance is equal or worse, you should revert the commit to preserve the history of the experiment but get back to the better code.

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep.
If they don't, discard.
And you're advancing the branch so that you can iterate.
If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run.
If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped.
You are autonomous.
If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes.
The loop runs until the human interrupts you, period.
