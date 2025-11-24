"""
Train Mamba baseline model (UNIDIRECTIONAL) once and evaluate on both validation and test sets.
"""

import os
import sys
import shutil
import numpy as np
import torch
import random
import pickle
import subprocess
from pathlib import Path

# Add the model_training directory to the path to ensure imports work
# regardless of where the script is run from
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

import pandas as pd
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

# Find the config file - it should be in the scripts directory
repo_root = script_dir.parent
args_path = repo_root / 'scripts' / 'minimal_mamba_args_unidirectional.yaml'
if not args_path.exists():
    # Fallback: try current directory
    args_path = Path('minimal_mamba_args_unidirectional.yaml')
    if not args_path.exists():
        # Another fallback: try scripts directory relative to current working directory
        args_path = Path('scripts') / 'minimal_mamba_args_unidirectional.yaml'
    
print(f"Loading configuration from: {args_path}")
if not args_path.exists():
    raise FileNotFoundError(f"Config file not found: {args_path}")
args = OmegaConf.load(str(args_path))

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(checkpoint_dir, model_name, eval_type, repo_root=None):
    """
    Run evaluate_sessions.py on the specified split (val or test) for this checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        model_name: Name of the model (for logging)
        eval_type: 'val' or 'test'
        repo_root: Root directory of the repository (auto-detected if None)
    
    Returns:
        Dictionary with metrics or None if evaluation failed
    """
    # Auto-detect repo root if not provided
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.absolute()
    else:
        repo_root = Path(repo_root)
    
    # Use minimal_evaluate.py (which is the actual evaluation script)
    eval_script = repo_root / 'scripts' / 'minimal_evaluate.py'
    data_dir = os.environ.get('DATA_DIR', str(repo_root / 'data' / 'hdf5_data_final'))
    csv_path = repo_root / 'data' / 't15_copyTaskData_description.csv'
    gpu_number = 0
    target_sessions = ["t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20"]
    
    # Check if the directory exists
    if not checkpoint_dir.exists():
        print(f"[{model_name}] ERROR: Directory does not exist: {checkpoint_dir}")
        return None

    # Check if args.yaml exists
    if not (checkpoint_dir / "args.yaml").exists():
        print(f"[{model_name}] WARNING: args.yaml not found in {checkpoint_dir}")
        return None

    # Check if best_checkpoint exists
    if not (checkpoint_dir / "best_checkpoint").exists():
        print(f"[{model_name}] WARNING: best_checkpoint not found in {checkpoint_dir}")
        return None

    cmd = [
        "python",
        str(eval_script),
        "--model_path",
        str(checkpoint_dir),
        "--data_dir",
        str(data_dir),
        "--csv_path",
        str(csv_path),
        "--eval_type",
        eval_type,
        "--gpu_number",
        str(gpu_number),
        "--sessions",
        *target_sessions,
    ]

    print(f"[{model_name}] Running {eval_type.upper()} evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run from the scripts directory - don't capture output so we can see errors
        work_dir = str(repo_root / 'scripts')
        result = subprocess.run(cmd, check=True, cwd=work_dir, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"[{model_name}] WARNING: {eval_type} evaluation failed with exit code: {e.returncode}")
        return None

    # The output CSV is saved in scripts/output directory (relative to where minimal_evaluate.py runs)
    output_dir = repo_root / 'scripts' / 'output'
    
    # Get files sorted by modification time (newest last)
    try:
        csv_files = sorted(
            output_dir.glob("phoneme_predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
    except Exception:
        csv_files = []
    
    if not csv_files:
        print(f"[{model_name}] WARNING: No phoneme_predictions_*.csv found in {output_dir}")
        return None

    latest_csv = csv_files[-1]
    print(f"[{model_name}] Processing {eval_type} metrics from: {latest_csv.name}")

    try:
        df = pd.read_csv(latest_csv)

        # Compute metrics
        avg_acc = df["trial_acc"].mean()
        total_ed = df["trial_ed"].sum()
        total_phoneme_length = df["true_phoneme"].str.split("-").str.len().sum()
        
        aggregate_per = (
            total_ed / total_phoneme_length if total_phoneme_length > 0 else float("inf")
        )
        avg_loss = df["trial_ctc_loss"].mean()

        print(f"[{model_name}] {eval_type.upper()} PER: {aggregate_per:.4f} | Acc: {avg_acc:.4f} | Loss: {avg_loss:.4f}")

        return {
            f"{eval_type}_avg_acc": avg_acc,
            f"{eval_type}_aggregate_per": aggregate_per,
            f"{eval_type}_avg_loss": avg_loss,
            f"{eval_type}_csv_path": str(latest_csv),
        }
    except Exception as e:
        print(f"[{model_name}] Error reading CSV: {e}")
        return None

# --- Main Training ---
print(f"\n{'='*80}")
print(f"Training Mamba Baseline Model (UNIDIRECTIONAL)")
print(f"{'='*80}\n")

# Set seed
if hasattr(args, 'seed'):
    seed = args.seed
else:
    seed = 10
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
    print(f"\nInitializing trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    
    print(f"Starting model training...")
    train_stats = trainer.train()
    
    # Extract best PER from training
    val_per_list = train_stats.get('val_PERs', [])
    best_per = np.min(val_per_list) if val_per_list else float('inf')
    
    # Save metrics for this run
    metrics_file = Path(args.output_dir) / 'training_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump({
            'metrics': train_stats,
            'args': args,
            'seed': seed,
        }, f)
    
    print(f"\nTraining completed!")
    print(f"Best PER (from training): {best_per:.6f}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Checkpoint saved in: {args.checkpoint_dir}")
    
    # Evaluate on validation
    print(f"\n{'='*80}")
    print(f"Evaluating on VALIDATION set")
    print(f"{'='*80}\n")
    repo_root = Path(__file__).parent.parent.absolute()
    val_metrics = evaluate_model(Path(args.checkpoint_dir), "MambaBaselineUni", 'val', repo_root=repo_root)
    
    # Evaluate on test
    print(f"\n{'='*80}")
    print(f"Evaluating on TEST set")
    print(f"{'='*80}\n")
    test_metrics = evaluate_model(Path(args.checkpoint_dir), "MambaBaselineUni", 'test', repo_root=repo_root)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}\n")
    print(f"Training Best PER: {best_per:.6f}")
    
    if val_metrics:
        print(f"\nValidation Results:")
        print(f"  PER: {val_metrics['val_aggregate_per']:.6f}")
        print(f"  Accuracy: {val_metrics['val_avg_acc']:.6f}")
        print(f"  Loss: {val_metrics['val_avg_loss']:.6f}")
    
    if test_metrics:
        print(f"\nTest Results:")
        print(f"  PER: {test_metrics['test_aggregate_per']:.6f}")
        print(f"  Accuracy: {test_metrics['test_avg_acc']:.6f}")
        print(f"  Loss: {test_metrics['test_avg_loss']:.6f}")
    
    # Save evaluation results
    results = {
        'training_best_per': best_per,
        'seed': seed,
        'checkpoint_dir': args.checkpoint_dir,
        'output_dir': args.output_dir,
    }
    if val_metrics:
        results.update(val_metrics)
    if test_metrics:
        results.update(test_metrics)
    
    results_file = Path(args.output_dir) / 'evaluation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nEvaluation results saved to: {results_file}")
    
except Exception as e:
    print(f"\nERROR during training: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print(f"Done!")
print(f"{'='*80}\n")

