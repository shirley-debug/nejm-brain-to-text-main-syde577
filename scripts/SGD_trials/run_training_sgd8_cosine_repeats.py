"""
SGD8: repeat the cosine scheduler (SGD5/Optuna best settings) multiple times
to measure repeatability of the cosine schedule.
"""

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import torch
import shutil
import random
import numpy as np
import pickle
import os
from pathlib import Path

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

NUM_RUNS = 3
SCHEDULER_TYPE = "cosine"
MOMENTUM = 0.88
WEIGHT_DECAY = 0.00446415850230071  # Best from Optuna trial 5


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_repeat(run_idx, base_args, original_output_dir, original_checkpoint_dir):
    args = base_args.copy()

    run_suffix = f"{SCHEDULER_TYPE}_run{run_idx+1}"
    args.output_dir = str(Path(original_output_dir).parent / f"{Path(original_output_dir).name}_{run_suffix}")
    args.checkpoint_dir = str(Path(original_checkpoint_dir).parent / f"{Path(original_checkpoint_dir).name}_{run_suffix}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)

    if hasattr(args, "seed"):
        seed_everything(args.seed + run_idx)

    trainer = BrainToTextDecoder_Trainer(args)

    bias_params = [p for name, p in trainer.model.named_parameters()
                   if ("gru.bias" in name or "out.bias" in name) and p.requires_grad]
    day_params = [p for name, p in trainer.model.named_parameters()
                  if "day_" in name and p.requires_grad]
    other_params = [p for name, p in trainer.model.named_parameters()
                    if "day_" not in name and "gru.bias" not in name and "out.bias" not in name
                    and p.requires_grad]

    weight_decay = WEIGHT_DECAY

    if day_params:
        param_groups = [
            {"params": bias_params, "weight_decay": 0, "group_type": "bias", "lr": args.lr_max},
            {"params": day_params, "lr": args.lr_max_day, "weight_decay": args.weight_decay_day, "group_type": "day_layer"},
            {"params": other_params, "group_type": "other", "lr": args.lr_max, "weight_decay": weight_decay},
        ]
    else:
        param_groups = [
            {"params": bias_params, "weight_decay": 0, "group_type": "bias", "lr": args.lr_max},
            {"params": other_params, "group_type": "other", "lr": args.lr_max, "weight_decay": weight_decay},
        ]

    trainer.optimizer = torch.optim.SGD(
        param_groups,
        momentum=MOMENTUM,
        nesterov=True,
    )

    scheduler_info = {
        "type": SCHEDULER_TYPE,
        "lr_max": args.lr_max,
        "lr_min": args.lr_min,
        "warmup_steps": args.lr_warmup_steps,
        "decay_steps": args.lr_decay_steps,
    }
    trainer.learning_rate_scheduler = trainer.create_cosine_lr_scheduler(trainer.optimizer)

    print(f"\n=== SGD8 Run {run_idx+1}/{NUM_RUNS} ({SCHEDULER_TYPE}) ===")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Momentum: {MOMENTUM}, Weight decay: {weight_decay}")
    print(f"Scheduler: {scheduler_info}")

    metrics = trainer.train()

    val_per_list = metrics.get("val_PERs", [])
    best_per = min(val_per_list) if val_per_list else float("inf")

    metrics_dir = Path(args.output_dir)
    metrics_file = metrics_dir / f"training_metrics_{run_suffix}.pkl"
    payload = {
        "metrics": metrics,
        "args": args,
        "optimizer_info": {
            "type": "SGD",
            "momentum": MOMENTUM,
            "nesterov": True,
            "lr_max": args.lr_max,
            "lr_max_day": args.lr_max_day,
            "weight_decay": weight_decay,
            "weight_decay_day": args.weight_decay_day,
            "scheduler_info": scheduler_info,
        },
    }
    with open(metrics_file, "wb") as f:
        pickle.dump(payload, f)

    parent_metrics_file = Path(original_output_dir).parent / f"training_metrics_{run_suffix}.pkl"
    with open(parent_metrics_file, "wb") as f:
        pickle.dump(payload, f)

    print(f"[Run {run_idx+1}] Best PER: {best_per:.6f}")
    print(f"[Run {run_idx+1}] Metrics saved to: {metrics_file}")
    print(f"[Run {run_idx+1}] Copy saved to: {parent_metrics_file}")

    return {
        "run": run_idx + 1,
        "best_per": best_per,
        "metrics_file": str(parent_metrics_file),
        "output_dir": args.output_dir,
    }


def main():
    args_path = "minimal_sgd_args.yaml"
    base_args = OmegaConf.load(args_path)

    original_output_dir = base_args.output_dir
    original_checkpoint_dir = base_args.checkpoint_dir

    print(f"Running SGD8 repeats (scheduler={SCHEDULER_TYPE}) with:")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Num runs: {NUM_RUNS}")
    print(f"  Base output dir: {original_output_dir}")

    results = []
    for run_idx in range(NUM_RUNS):
        results.append(run_single_repeat(run_idx, base_args, original_output_dir, original_checkpoint_dir))

    print("\n=== SGD8 Summary ===")
    results_sorted = sorted(results, key=lambda x: x["best_per"])
    for r in results_sorted:
        print(f"Run {r['run']}: PER={r['best_per']:.6f} | metrics={r['metrics_file']}")
    best = results_sorted[0]
    print(f"\nBest run: Run {best['run']} with PER={best['best_per']:.6f}")


if __name__ == "__main__":
    main()

