"""
Evaluate all SGD models in minimal_sgd folder on both VALIDATION and TEST splits.
"""

import os
import numpy as np
import subprocess
from pathlib import Path
import pandas as pd

# --- Configuration ---
os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

# 1. DIRECTORY TO RUN THE SCRIPT FROM (Crucial for relative paths to data CSVs)
WORK_DIR = "/kaggle/working/nejm-brain-to-text/model_training/"

# 2. ROOT DIRECTORY WHERE SGD MODELS ARE SAVED
MODELS_ROOT = Path("/kaggle/working/nejm-brain-to-text/model_training/trained_models/minimal_sgd")

TARGET_SESSIONS = ["t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20"]

def evaluate_model(checkpoint_dir, model_name, eval_type):
    """
    Run evaluate_sessions.py on the specified split (val or test) for this checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        model_name: Name of the model (for logging)
        eval_type: 'val' or 'test'
    
    Returns:
        Dictionary with metrics or None if evaluation failed
    """
    eval_script = "evaluate_sessions.py"
    data_dir = os.environ['DATA_DIR']
    gpu_number = 0

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
        eval_script,
        "--model_path",
        str(checkpoint_dir),  # Passing the FOLDER, not the file
        "--data_dir",
        data_dir,
        "--eval_type",
        eval_type,
        "--gpu_number",
        str(gpu_number),
        "--sessions",
        *TARGET_SESSIONS,
    ]

    print(f"[{model_name}] Running {eval_type.upper()} evaluation...")
    
    try:
        # cwd=WORK_DIR ensures relative paths in evaluate_sessions.py work correctly
        subprocess.run(cmd, check=True, cwd=WORK_DIR, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[{model_name}] WARNING: {eval_type} evaluation failed with error: {e}")
        return None

    # The output will be inside the WORK_DIR/output
    output_dir = Path(WORK_DIR) / "output"
    
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

def find_checkpoint_directories(models_root):
    """
    Find all checkpoint directories in the models root.
    Returns a list of (checkpoint_dir, model_name) tuples.
    """
    checkpoint_dirs = []
    
    if not models_root.exists():
        print(f"ERROR: Models root directory does not exist: {models_root}")
        return checkpoint_dirs
    
    # Look for directories that contain args.yaml and best_checkpoint
    for item in models_root.iterdir():
        if item.is_dir():
            if (item / "args.yaml").exists() and (item / "best_checkpoint").exists():
                checkpoint_dirs.append((item, item.name))
    
    # Sort by name for consistent ordering
    checkpoint_dirs.sort(key=lambda x: x[1])
    
    return checkpoint_dirs

# --- Main Execution ---
print(f"\n{'='*80}")
print(f"Evaluating SGD models on VALIDATION and TEST splits")
print(f"Models root: {MODELS_ROOT}")
print(f"{'='*80}\n")

# Find all checkpoint directories
checkpoint_dirs = find_checkpoint_directories(MODELS_ROOT)

if not checkpoint_dirs:
    print(f"ERROR: No checkpoint directories found in {MODELS_ROOT}")
    print("Expected directories containing args.yaml and best_checkpoint")
    exit(1)

print(f"Found {len(checkpoint_dirs)} checkpoint directories:\n")
for _, name in checkpoint_dirs:
    print(f"  - {name}")

results = []

# Evaluate each model on both val and test
for checkpoint_dir, model_name in checkpoint_dirs:
    print(f"\n{'#'*80}")
    print(f"# Processing: {model_name}")
    print(f"{'#'*80}")
    
    result_entry = {
        'model_name': model_name,
        'checkpoint_path': str(checkpoint_dir)
    }
    
    # Evaluate on validation
    val_metrics = evaluate_model(checkpoint_dir, model_name, 'val')
    if val_metrics:
        result_entry.update(val_metrics)
        result_entry['val_status'] = "Success"
    else:
        result_entry['val_status'] = "Failed"
        result_entry['val_aggregate_per'] = np.nan
        result_entry['val_avg_acc'] = np.nan
        result_entry['val_avg_loss'] = np.nan
    
    # Evaluate on test
    test_metrics = evaluate_model(checkpoint_dir, model_name, 'test')
    if test_metrics:
        result_entry.update(test_metrics)
        result_entry['test_status'] = "Success"
    else:
        result_entry['test_status'] = "Failed"
        result_entry['test_aggregate_per'] = np.nan
        result_entry['test_avg_acc'] = np.nan
        result_entry['test_avg_loss'] = np.nan
    
    results.append(result_entry)

# --- Print Summary Tables ---
print(f"\n{'='*80}")
print(f"EVALUATION SUMMARY")
print(f"{'='*80}\n")

# Sort by model name
results_sorted = sorted(results, key=lambda x: x['model_name'])
print(results_sorted)

# create %%writefile /kaggle/working/nejm-brain-to-text/model_training/eval_sgd_models_val_test.py
# run with !python /kaggle/working/nejm-brain-to-text/model_training/eval_sgd_models_val_test.py
