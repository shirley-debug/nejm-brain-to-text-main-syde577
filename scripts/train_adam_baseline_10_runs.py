"""
Train Adam baseline model 10 times to measure variance.
Each run gets a unique checkpoint directory and saves metrics separately.
"""

import os
import shutil
import numpy as np
import torch
import random
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
from scipy import stats

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

args_path = 'minimal_baseline_args.yaml'
print(f"Loading configuration from: {args_path}")
base_args = OmegaConf.load(args_path)

original_output_dir = base_args.output_dir
original_checkpoint_dir = base_args.checkpoint_dir

NUM_RUNS = 10
BASE_NAME = "adambaseline"

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on_test_for_run(checkpoint_dir, run_name):
    """
    Run evaluate_sessions.py on the test split for this checkpoint and return
    three metrics: avg trial acc, aggregate PER, avg loss.
    """
    eval_script = "/kaggle/working/nejm-brain-to-text/model_training/evaluate_sessions.py"
    data_dir = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"
    eval_type = "test"
    gpu_number = 0

    # Sessions to evaluate
    target_sessions = ["t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20"]

    cmd = [
        "python",
        eval_script,
        "--model_path",
        checkpoint_dir,
        "--data_dir",
        data_dir,
        "--eval_type",
        eval_type,
        "--gpu_number",
        str(gpu_number),
        "--sessions",
        *target_sessions,
    ]

    print(f"[{run_name}] Running test evaluation:")
    print(" ", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{run_name}] WARNING: evaluation failed with error: {e}")
        return None

    # Find the latest phoneme_predictions_*.csv in output directory
    output_dir = Path("/kaggle/working/nejm-brain-to-text/model_training/output")
    csv_files = sorted(
        output_dir.glob("phoneme_predictions_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not csv_files:
        print(f"[{run_name}] WARNING: No phoneme_predictions_*.csv found in {output_dir}")
        return None

    latest_csv = csv_files[-1]
    print(f"[{run_name}] Using evaluation CSV: {latest_csv}")

    df = pd.read_csv(latest_csv)

    # Compute three test metrics:
    test_avg_acc = df["trial_acc"].mean()
    total_ed = df["trial_ed"].sum()
    total_phoneme_length = df["true_phoneme"].str.split("-").str.len().sum()
    test_aggregate_per = (
        total_ed / total_phoneme_length if total_phoneme_length > 0 else float("inf")
    )
    test_avg_loss = df["trial_ctc_loss"].mean()

    print(f"[{run_name}] Test avg trial acc: {test_avg_acc:.4f}")
    print(f"[{run_name}] Test aggregate PER: {test_aggregate_per:.4f}")
    print(f"[{run_name}] Test avg loss: {test_avg_loss:.4f}")

    return {
        "test_avg_acc": test_avg_acc,
        "test_aggregate_per": test_aggregate_per,
        "test_avg_loss": test_avg_loss,
        "test_csv": str(latest_csv),
    }

# --- Run Training 10 Times ---
results = []

print(f"\n{'='*80}")
print(f"Training {BASE_NAME} model {NUM_RUNS} times")
print(f"{'='*80}\n")

for run_idx in range(NUM_RUNS):
    print(f"\n{'#'*80}")
    print(f"# RUN {run_idx + 1}/{NUM_RUNS}")
    print(f"{'#'*80}\n")
    
    # Create a copy of args for this run
    args = base_args.copy()
    
    # Set unique output and checkpoint directories
    run_name = f"{BASE_NAME}_run{run_idx + 1}"
    args.output_dir = f"{original_output_dir}_{run_name}"
    args.checkpoint_dir = f"{original_checkpoint_dir}_{run_name}"
    
    # Set seed for this run (different seed each time)
    if hasattr(args, 'seed'):
        seed = args.seed + run_idx
    else:
        seed = 10 + run_idx
    seed_everything(seed)
    args.seed = seed
    
    # Clean up old directories
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
        print(f"Removing existing checkpoint directory: {args.checkpoint_dir}")
        shutil.rmtree(args.checkpoint_dir)
    
    try:
        # Initialize and Run Trainer
        print(f"\n[Run {run_idx + 1}] Initializing trainer...")
        trainer = BrainToTextDecoder_Trainer(args)
        
        print(f"[Run {run_idx + 1}] Starting model training...")
        train_stats = trainer.train()
        
        # Extract best PER
        val_per_list = train_stats.get('val_PERs', [])
        best_per = np.min(val_per_list) if val_per_list else float('inf')
        
        # Save metrics for this run
        metrics_file = Path(args.output_dir) / 'training_metrics.pkl'
        with open(metrics_file, 'wb') as f:
            pickle.dump({
                'metrics': train_stats,
                'args': args,
                'run_number': run_idx + 1,
                'seed': seed,
            }, f)
        
        print(f"\n[Run {run_idx + 1}] Training completed!")
        print(f"[Run {run_idx + 1}] Best PER: {best_per:.6f}")
        print(f"[Run {run_idx + 1}] Metrics saved to: {metrics_file}")
        print(f"[Run {run_idx + 1}] Checkpoint saved in: {args.checkpoint_dir}")
        
        # Run test evaluation for this run
        eval_metrics = evaluate_on_test_for_run(args.checkpoint_dir, run_name)

        # Store result (including test metrics if available)
        result_entry = {
            'run': run_idx + 1,
            'seed': seed,
            'best_per': best_per,
            'output_dir': args.output_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'metrics_file': str(metrics_file)
        }
        if eval_metrics is not None:
            result_entry.update(eval_metrics)

        results.append(result_entry)
        
    except Exception as e:
        print(f"\n[Run {run_idx + 1}] ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'run': run_idx + 1,
            'seed': seed,
            'best_per': float('inf'),
            'error': str(e)
        })
    
    print(f"\n[Run {run_idx + 1}] {'='*60}\n")

# --- Print Summary ---
print(f"\n{'='*80}")
print(f"TRAINING SUMMARY - {NUM_RUNS} RUNS")
print(f"{'='*80}\n")

# Sort by best PER
results_sorted = sorted(results, key=lambda x: x['best_per'])

print(f"{'Run':<6} {'Seed':<8} {'Best PER':<12} {'Test PER':<12} {'Test Acc':<12} {'Test Loss':<12} {'Status':<20}")
print(f"{'-'*80}")
for result in results_sorted:
    run = result['run']
    seed = result['seed']
    best_per = result['best_per']
    test_per = result.get('test_aggregate_per', float('nan'))
    test_acc = result.get('test_avg_acc', float('nan'))
    test_loss = result.get('test_avg_loss', float('nan'))
    if 'error' in result:
        status = f"ERROR: {result['error'][:30]}"
    else:
        status = "✓ Success"
    print(f"{run:<6} {seed:<8} {best_per:<12.6f} {test_per:<12.6f} {test_acc:<12.6f} {test_loss:<12.6f} {status:<20}")

print(f"\n{'='*80}")
print(f"STATISTICS:")
print(f"{'='*80}")
valid_results = [r for r in results if 'error' not in r and r['best_per'] != float('inf')]
if valid_results:
    pers = np.array([r['best_per'] for r in valid_results])
    n = len(pers)
    mean_per = np.mean(pers)
    std_per = np.std(pers, ddof=1)  # Sample standard deviation (ddof=1 for unbiased estimate)
    median_per = np.median(pers)
    
    # Calculate 95% Confidence Interval using bootstrap method
    ci_low, ci_high = stats.bootstrap(
        (pers,),
        np.mean,
        confidence_level=0.95
    ).confidence_interval
    
    print(f"  Total runs: {NUM_RUNS}")
    print(f"  Successful runs: {len(valid_results)}")
    print(f"  Best PER: {min(pers):.6f} (Run {results_sorted[0]['run']})")
    print(f"  Worst PER: {max(pers):.6f}")
    print(f"  Mean PER: {mean_per:.6f} ± {std_per:.6f} (std)")
    print(f"  95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  Median PER: {median_per:.6f}")
else:
    print(f"  No successful runs to calculate statistics")

print(f"\n{'='*80}")
print(f"All checkpoints saved with prefix: {BASE_NAME}_runX")
print(f"{'='*80}\n")

# Save summary to file
summary_file = Path(original_output_dir).parent / f'{BASE_NAME}_10_runs_summary.pkl'
if valid_results:
    pers = np.array([r['best_per'] for r in valid_results])
    n = len(pers)
    mean_per = np.mean(pers)
    std_per = np.std(pers, ddof=1)
    
    # Calculate CI again for saving using bootstrap
    ci_low, ci_high = stats.bootstrap(
        (pers,),
        np.mean,
        confidence_level=0.95
    ).confidence_interval
    
    stats_dict = {
        'total_runs': NUM_RUNS,
        'successful_runs': len(valid_results),
        'best_per': min(pers),
        'worst_per': max(pers),
        'mean_per': mean_per,
        'std_per': std_per,
        'median_per': np.median(pers),
        'ci_95_lower': ci_low,
        'ci_95_upper': ci_high,
    }
else:
    stats_dict = {}

with open(summary_file, 'wb') as f:
    pickle.dump({
        'results': results_sorted,
        'statistics': stats_dict
    }, f)
print(f"Summary saved to: {summary_file}")

