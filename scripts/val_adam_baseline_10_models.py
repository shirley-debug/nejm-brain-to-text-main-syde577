"""
Evaluate the 10 previously trained Adam baseline models on the VALIDATION split.-
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

# 2. EXACT ROOT WHERE CHECKPOINTS ARE SAVED
OUTPUT_ROOT = Path("/kaggle/working/output_dir")

NUM_RUNS = 10
BASE_NAME = "adambaseline"
TARGET_SESSIONS = ["t15.2023.08.13", "t15.2023.08.18", "t15.2023.08.20"]

def evaluate_on_val_for_run(run_dir, run_name):
    """
    Run evaluate_sessions.py on the VAL split for this checkpoint directory.
    """
    eval_script = "evaluate_sessions.py"
    data_dir = os.environ['DATA_DIR']
    eval_type = "val" 
    gpu_number = 0

    # Check if the directory exists
    if not run_dir.exists():
        print(f"[{run_name}] CRITICAL ERROR: Directory does not exist: {run_dir}")
        return None

    # Check if args.yaml exists (Debug check)
    if not (run_dir / "args.yaml").exists():
        print(f"[{run_name}] WARNING: args.yaml not found in {run_dir}. Script might fail.")

    cmd = [
        "python",
        eval_script,
        "--model_path",
        str(run_dir), # Passing the FOLDER, not the file
        "--data_dir",
        data_dir,
        "--eval_type",
        eval_type,
        "--gpu_number",
        str(gpu_number),
        "--sessions",
        *TARGET_SESSIONS,
    ]

    print(f"[{run_name}] Running VAL evaluation...")
    
    try:
        # cwd=WORK_DIR ensures relative paths in evaluate_sessions.py work correctly
        subprocess.run(cmd, check=True, cwd=WORK_DIR)
    except subprocess.CalledProcessError as e:
        print(f"[{run_name}] WARNING: evaluation failed with error: {e}")
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
        print(f"[{run_name}] WARNING: No phoneme_predictions_*.csv found in {output_dir}")
        return None

    latest_csv = csv_files[-1]
    print(f"[{run_name}] Processing metrics from: {latest_csv.name}")

    try:
        df = pd.read_csv(latest_csv)

        # Compute validation metrics
        val_avg_acc = df["trial_acc"].mean()
        total_ed = df["trial_ed"].sum()
        total_phoneme_length = df["true_phoneme"].str.split("-").str.len().sum()
        
        val_aggregate_per = (
            total_ed / total_phoneme_length if total_phoneme_length > 0 else float("inf")
        )
        val_avg_loss = df["trial_ctc_loss"].mean()

        print(f"[{run_name}] Val PER: {val_aggregate_per:.4f} | Acc: {val_avg_acc:.4f}")

        return {
            "val_avg_acc": val_avg_acc,
            "val_aggregate_per": val_aggregate_per,
            "val_avg_loss": val_avg_loss,
            "csv_path": str(latest_csv),
        }
    except Exception as e:
        print(f"[{run_name}] Error reading CSV: {e}")
        return None

# --- Main Execution Loop ---
results = []

print(f"\n{'='*80}")
print(f"Evaluating {BASE_NAME} models on VALIDATION split ({NUM_RUNS} runs)")
print(f"{'='*80}\n")

for run_idx in range(NUM_RUNS):
    run_num = run_idx + 1
    run_name = f"{BASE_NAME}_run{run_num}"
    
    print(f"\n{'#'*40}")
    print(f"# Processing RUN {run_num}/{NUM_RUNS}")
    print(f"{'#'*40}")

    # Construct path to the FOLDER: /kaggle/working/output_dir/checkpoints_adambaseline_runX
    folder_name = f"checkpoints_{run_name}" 
    current_checkpoint_dir = OUTPUT_ROOT / folder_name
    
    print(f"[{run_name}] Targeting directory: {current_checkpoint_dir}")

    # Run Evaluation
    metrics = evaluate_on_val_for_run(current_checkpoint_dir, run_name)
    
    result_entry = {
        'run': run_num,
        'checkpoint_path': str(current_checkpoint_dir)
    }
    
    if metrics:
        result_entry.update(metrics)
        result_entry['status'] = "Success"
    else:
        result_entry['status'] = "Failed"
        result_entry['val_aggregate_per'] = np.nan
        result_entry['val_avg_acc'] = np.nan

    results.append(result_entry)

# --- Print Summary ---
print(f"\n{'='*80}")
print(f"VALIDATION EVALUATION SUMMARY")
print(f"{'='*80}\n")

# Sort by Run number
results_sorted = sorted(results, key=lambda x: x['run'])
print(results_sorted)

# create %%writefile /kaggle/working/nejm-brain-to-text/model_training/eval_val_adam_10_runs.py
# run with !python /kaggle/working/nejm-brain-to-text/model_training/eval_val_adam_10_runs.py