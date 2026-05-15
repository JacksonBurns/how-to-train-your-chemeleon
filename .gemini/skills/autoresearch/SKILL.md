---
name: autoresearch
description:
   You are an AI assistant whose job is to autonomously perform research (an "experiment") towards improving the `CheMeleon` foundation model, eventually arriving at `CheMeleon2`.
---

# autoresearch

You are an AI assistant whose job is to autonomously perform research (an "experiment") towards improving the `CheMeleon` foundation model, eventually arriving at `CheMeleon2`.

You will use the conda environment `httyc` to execute all code in this skill, except for `evaluate.py` which will be run in the `polaris` environment.
Check before execution that you can use these environments.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`).
2. **Create the branch**: checkout a new branch with a name derived from the current branch, i.e. `git checkout -b <current_name>-autoresearch-mar5`
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` and `GEMINI.md` — repository context.
   - `pretraining/` - the files you modify.
   Model architecture, optimizer, training loop, etc. are all defined in here.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on this 8-GPU machine named `mithrim` and is launched from the script `pretraining/train.py`.
This script requires 2 inputs: the data source (you should ask the user for this) and the output directory (which should just be set to `chemeleon2-autoresearch-mar5` or i.e. match the branch name)

**What you CAN do:**

 - Modify model architecture by changing `pretraining/config.py`, `pretraining/multiweight_message_passing.py`, `pretraining/random_dropout_mse.py`, and `pretraining/train.py`.Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

> **NOTE**
> You are optimizing the architecture on a subset of the available training data - the true scale is about 50x larger, so some parameters like batch size may not be as pertinent.

**What you CANNOT do:**

 - Modify the evaluation code (`evaluate.py`) - except to make it compatible with the model loading logic, e.g. aggregation function, according to your changes - or the utility code in `now.py` and `split.py`.
 - Modify training or validation data - this should remain as exactly what the user requests.
 - Install new packages or add dependencies. You can only use what's already installed in the `httyc` conda environment.

**The goal is simple: get the best evaluation performance.** Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful performance gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Exploration diversity**: Don't just tweak hyperparameters given in `config.py`.
Alternate between categories: architecture (depth, width, attention), optimization (LR, batch size, schedules), regularization, and simplification (removing components).
If your last 3 experiments were all in the same category, try a different one.
Edit the training routine and model architecture in ways not reflected in the `config.py` file - anything to improve the model.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the `pretraining/train.py` script finishes, it prints a path to the trained model like this:

```
Best model file: chemeleon2_preview/2026-05-14_15-53-28/checkpoints/epoch=5-step=1000.ckpt
```

You should then run the `pretraining/extract_mp.py` file to retrieve the message passing weights (the actual foundation model).
This script will also print the filepath to the new weights.
Example:

```
$python extract_mp.py chemeleon2/2026-05-15_15-20-51/checkpoints/epoch\=0-step\=750.ckpt 
/home/jwburns/miniforge3/envs/httyc/lib/python3.13/site-packages/cuik_molmaker/mol_features.py:10: output will be written to '/home/jwburns/how-to-train-your-chemeleon/pretraining/chemeleon2/2026-05-15_15-20-51/checkpoints/epoch=0-step=750_mp.pt'
```

Finally, the evaluation script prints a series of benchmarks names and their corresponding tables showing important performance metrics.
You should read through these (they are printed at the end of the script) to understand how well the pretraining worked.
You should devise a way to summarize these metrics into a single scalar that is monotonic for you to focus on improving (make it clear in your experiments how you do so by setting the header in the output file to an appropriate description).

```
$ CUDA_VISIBLE_DEVICES=3 conda run -n polaris python evaluate.py chemeleon2/2026-05-15_15-20-51/checkpoints/epoch\=0-step\=750_mp.pt 
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and these columns:

```
commit	performance (description)	status	description
```

1. git commit hash (short, 7 chars)
2. evaluation performance achieved (e.g. 1.234567) — use a blank for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	performance (description)	status	description
a1b2c3d	0.997900	keep	baseline
b2c3d4e	0.993200	keep	increase LR to 0.04
c3d4e5f	1.005000	discard	switch to GeLU activation
d4e5f6g		0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `pretraining/config.py`, `pretraining/multiweight_message_passing.py`, `pretraining/random_dropout_mse.py`, and `pretraining/train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment by executing `train.py`, then `extract_mp.py`, and finally `evaluate.py` - this last script should always be run with `CUDA_VISIBLE_DEVICES=3 python evaluate.py ...` to ensure it uses only one GPU.
 Example: `CUDA_VISIBLE_DEVICES=3 conda run -n polaris python evaluate.py chemeleon2/2026-05-15_15-20-51/checkpoints/epoch\=0-step\=750_mp.pt`
 (redirect everything for all scripts — do NOT use tee or let output flood your context)
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
