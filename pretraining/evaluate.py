
# evaluate.py
#
# runs the model on the benchmarks
import os
import sys
from pathlib import Path

import numpy as np
import polaris as po
import torch
from astartes import train_test_split
from chemprop.data import LazyMoleculeDatapoint, CuikmolmakerDataset, build_dataloader
from chemprop.featurizers import CuikmolmakerMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN, UnscaleTransform, BinaryClassificationFFN, BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from polaris.utils.types import TargetType

from multiweight_message_passing import MultiweightMessagePassing

from config import FEATURIZER, MP_TYPE


if __name__ == "__main__":
    v = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if v is None or v != '3':
        print("Set CUDA_VISIBLE_DEVICES=3 before running this script, i.e. CUDA_VISIBLE_DEVICES= python evaluate.py path.pt")
        exit(1)
    
    try:
        mp_path = Path(sys.argv[1])
    except:
        print("usage: python evaluate.py /path/to/message_passing_mp.pt")
        exit(1)

    polaris_benchmarks = (
        "polaris/pkis2-ret-wt-cls-v2",
        "polaris/adme-fang-solu-1",
        "tdcommons/clearance-hepatocyte-az",
        "tdcommons/bbb-martins",
    )
    random_seed = 42
    summary_scores = []
    result_str = ""
    for benchmark_name in (polaris_benchmarks):
        # load the benchmarking data
        benchmark = po.load_benchmark(benchmark_name)
        smiles_col = list(benchmark.input_cols)[0]
        target_cols = list(benchmark.target_cols)
        train, test = benchmark.get_train_test_split()
        train_df, test_df = train.as_dataframe(), test.as_dataframe()
        task_type = benchmark.target_types[target_cols[0]]

        targets = train_df[target_cols]
        targets = targets.fillna(targets.mean(axis=0)).to_numpy()

        #########################################
        # MODEL LOADING LOGIC - you can modify this as needed to be compatible with your model changes, but the evaluation logic should remain the same
        #########################################
        featurizer = CuikmolmakerMolGraphFeaturizer(FEATURIZER)
        _mp = torch.load(mp_path, weights_only=True)
        if MP_TYPE == "UNTIED":
            mp = MultiweightMessagePassing(**_mp["hyper_params"])
        else:
            mp = BondMessagePassing(**_mp["hyper_params"])
        mp.load_state_dict(_mp["state_dict"])
        agg = MeanAggregation()
        #########################################
        # END OF MODEL LOADING LOGIC
        #########################################
        
        hidden_size = mp.output_dim

        # typical chemprop training
        train_idxs, val_idxs = train_test_split(
            np.arange(len(targets)),
            train_size=0.80,
            test_size=0.20,
            random_state=random_seed,
        )
        train_data = [
            LazyMoleculeDatapoint(smi, y=y)
            for smi, y in zip(
                train_df[smiles_col].iloc[train_idxs], targets[train_idxs]
            )
        ]
        val_data = [
            LazyMoleculeDatapoint(smi, y=y)
            for smi, y in zip(
                train_df[smiles_col].iloc[val_idxs], targets[val_idxs]
            )
        ]
        test_data = [LazyMoleculeDatapoint(s) for s in test_df[smiles_col]]
        train_dataset = CuikmolmakerDataset(train_data, featurizer)
        val_dataset = CuikmolmakerDataset(val_data, featurizer)
        test_dataset = CuikmolmakerDataset(test_data, featurizer)
        scaler = None
        if task_type == TargetType.REGRESSION:
            scaler = train_dataset.normalize_targets()
            val_dataset.normalize_targets(scaler)
        train_dataloader = build_dataloader(train_dataset, num_workers=1)
        val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False)
        test_dataloader = build_dataloader(
            test_dataset, num_workers=1, shuffle=False
        )
        output_transform = (
            UnscaleTransform.from_standard_scaler(scaler)
            if scaler is not None
            else torch.nn.Identity()
        )
        fnn = (
            RegressionFFN(
                output_transform=output_transform,
                input_dim=hidden_size,
                hidden_dim=512,
            )
            if task_type == TargetType.REGRESSION
            else BinaryClassificationFFN(
                output_transform=output_transform,
                input_dim=hidden_size,
                hidden_dim=512,
            )
        )
        model = MPNN(mp, agg, fnn)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=False,
                patience=5,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                dirpath="tuning_checkpoints",
            ),
        ]
        trainer = Trainer(
            max_epochs=50,
            logger=False,
            log_every_n_steps=1,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(f"Reloading best model from checkpoint file: {ckpt_path}")
        model = MPNN.load_from_checkpoint(ckpt_path)
        trainer = Trainer(logger=False, enable_model_summary=False, enable_progress_bar=False)
        predictions = (
            torch.vstack(trainer.predict(model, test_dataloader))
            .numpy(force=True)
            .flatten()
        )
        result_str += benchmark_name + "\n"
        if task_type == TargetType.CLASSIFICATION:
            # we don't 'calibrate models' round these parts...
            results = benchmark.evaluate(predictions > 0.5, predictions).results
            result_str += results.to_markdown() + "\n"
        elif task_type == TargetType.REGRESSION:
            results = benchmark.evaluate(predictions).results
            result_str += results.to_markdown() + "\n"
        
        metric_map = {
            row["Metric"]: row["Score"]
            for _, row in results.iterrows()
        }

        # benchmark-specific normalized scalar
        if benchmark_name == "polaris/pkis2-ret-wt-cls-v2":
            score = metric_map["pr_auc"]

        elif benchmark_name == "polaris/adme-fang-solu-1":
            score = max(0.0, metric_map["pearsonr"])

        elif benchmark_name == "tdcommons/clearance-hepatocyte-az":
            score = max(0.0, metric_map["spearmanr"])

        elif benchmark_name == "tdcommons/bbb-martins":
            score = metric_map["roc_auc"]

        summary_scores.append(score)
        Path(ckpt_path).unlink()

    summary_metric = float(np.mean(summary_scores))

    print(f"\nSUMMARY_METRIC: {summary_metric:.6f}")

    with open("results.txt", "w") as file:
        file.write(result_str)
        file.write(f"\nSUMMARY_METRIC: {summary_metric:.6f}\n")
