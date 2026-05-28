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
   - `pretraining/train.py` - the file you modify.
   Model architecture, optimizer, training loop, etc. are all defined in here.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on this 8-GPU machine named `mithrim` and is launched from the script `train.py` within the `pretraining` directory.
This script requires 2 inputs: the pre-split training and validation data (you should ask the user for this) and the output directory (which should just be set to `chemeleon2-autoresearch-mar5` or i.e. match the branch name)

**What you CAN do:**

 - Modify model architecture model architecture, optimizer, hyperparameters, training loop, batch size, model size, by changing `pretraining/train.py`.
 Everything is fair game here, but the code must run without crashing.

> **NOTE**
> You are optimizing the architecture on a subset of the available training data - the true scale is about 50x larger, so some parameters like batch size may not be as pertinent.

**What you CANNOT do:**

 - Modify the utility code in `now.py` and `split.py`.
 These are not related to the model or training, and should be left alone.
 - Install new packages or add dependencies. You can only use what's already installed in the `httyc` conda environment.

**The goal is simple: get the best evaluation performance.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size.
The only constraint is that the code runs without crashing.
You can and should implement new classes based on the presented examples to add new functionality, but the code must run without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful performance gains, but it should not blow up dramatically.
Arbitrarily large increases may cause the run to crash.
If a run crashes due to OOM, you can try to fix it by reducing batch size or model size, but if the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.
Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.
When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.
A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.
An improvement of ~0 but much simpler code? Keep.

**Exploration diversity**: Don't just tweak hyperparameters like learning rate and hidden size.
Edit the training routine and model architecture in substantive ways.
Try adding new components, removing components, changing the way the model works in more fundamental ways.
Try things that are radically different from the baseline, not just small tweaks.
The goal is to explore the space of architectures and training methods as broadly as possible to find the best one, so you should be trying a wide variety of things, not just small tweaks to the same architecture.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Experiment Commands and Output Format

This should be run from the `pretraining` directory within this repository.
It will redirect all output to `train_output.log` which you can read after the run finishes.
Note that each of this command can take a *long time*.
You should run it in the foreground and wait for the output - DO NOT send them to background, just wait patiently for them to exit and then check status, read output files, etc.

To run the experiment run the training script, which should look like this:

```bash
conda run --no-capture-output -n httyc python train.py /path/to/data_split chemeleon2 &> train_output.log
```

**CRITICAL**: Run this command in the foreground and retain the `&>` redirection to capture all output to the log file - do not run it in the background or flood your context window with training output.

The last line of the file (read it with `tail`) tells you what the validation performance was for that training run.
Example:

```bash
Best model validation mse: 0.28369
```

You should extract this and record it in the `results.tsv` file along with a short description of what you changed in this experiment.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and these columns:

```
commit	validation mse	status	description
```

1. git commit hash (short, 7 chars)
2. validation MSE (the `val/mse` metric from the training log) — use a blank for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	validation mse	status	description
a1b2c3d	0.283693	keep	baseline
b2c3d4e	0.226891	keep	increase LR to 0.04
c3d4e5f	0.345551	discard	switch to GeLU activation
d4e5f6g			crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `pretraining/train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment by following the instructions in the [Experiment Commands and Output Format](#experiment-commands-and-output-format) to record the results.
5. If the run crashed at any stage, read the Python stack trace and attempt a fix.
If you can't get things to work after more than two attempts, give up.
6. Record the results in the tsv (NOTE: commit the results.tsv file to git can track it)
7. If performance improved, you "advance" the branch, keeping the git commit
8. If performance is equal or worse, you should revert the commit to preserve the history of the experiment but get back to the better code.

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
