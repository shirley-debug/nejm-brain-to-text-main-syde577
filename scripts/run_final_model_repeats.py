"""
Final Model Training: Run BiGRUDecoder with SGD + all optimizations 3 times
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_training'))

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import torch
import shutil
import random
import numpy as np
import pickle
import subprocess
from pathlib import Path

# Configuration
NUM_RUNS = 3
BASE_SEED = 10


def seed_everything(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_repeat(run_idx, base_args, original_output_dir, original_checkpoint_dir):
    """Run a single training run with a unique seed"""
    
    args = base_args.copy()
    
    # Create unique directories for this run
    run_suffix = f"run{run_idx+1}"
    args.output_dir = str(Path(original_output_dir).parent / f"{Path(original_output_dir).name}_{run_suffix}")
    args.checkpoint_dir = str(Path(original_checkpoint_dir).parent / f"{Path(original_checkpoint_dir).name}_{run_suffix}")
    
    # Clean up existing directories
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    
    # Set seed for this run
    run_seed = BASE_SEED - run_idx
    args.seed = run_seed
    seed_everything(run_seed)
    
    print(f"=== Run {run_idx+1}/{NUM_RUNS} ===")
    print(f"Seed: {run_seed}")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Architecture: {args.model.get('architecture', 'GRUDecoder')}")
    print(f"Optimizer: {args.optimizer_type}")
    print(f"LR Max: {args.lr_max}, LR Min: {args.lr_min}")
    print(f"Momentum: {args.momentum}, Weight Decay: {args.weight_decay}")
    print(f"CTC Weight: {args.loss.ctc_weight}, Blank Penalty: {args.loss.blank_penalty}")
    if hasattr(args.dataset.data_transforms, 'time_masking'):
        print(f"Time Masking: {args.dataset.data_transforms.time_masking}, Feature Masking: {args.dataset.data_transforms.feature_masking}")
    print(f"{'='*80}\n")
    
    # Create trainer
    trainer = BrainToTextDecoder_Trainer(args)
    
    # Train the model
    metrics = trainer.train()
    
    # Get best validation PER
    val_per_list = metrics.get("val_PERs", [])
    best_val_per = min(val_per_list) if val_per_list else float("inf")
    
    # Save training metrics
    metrics_dir = Path(args.output_dir)
    metrics_file = metrics_dir / f"training_metrics_{run_suffix}.pkl"
    payload = {
        "metrics": metrics,
        "args": args,
        "run_info": {
            "run_idx": run_idx + 1,
            "seed": run_seed,
            "best_val_PER": best_val_per,
        }
    }
    with open(metrics_file, "wb") as f:
        pickle.dump(payload, f)
    
    # Also save a copy in parent directory
    parent_metrics_file = Path(original_output_dir).parent / f"training_metrics_{run_suffix}.pkl"
    with open(parent_metrics_file, "wb") as f:
        pickle.dump(payload, f)
    
    print(f"\n[Run {run_idx+1}] Training Complete!")
    print(f"[Run {run_idx+1}] Best Val PER: {best_val_per:.6f}")
    print(f"[Run {run_idx+1}] Metrics saved to: {metrics_file}")
    print(f"[Run {run_idx+1}] Copy saved to: {parent_metrics_file}")
    
    return {
        "run": run_idx + 1,
        "seed": run_seed,
        "best_val_per": best_val_per,
        "metrics_file": str(parent_metrics_file),
        "checkpoint_dir": args.checkpoint_dir,
        "output_dir": args.output_dir,
    }


def evaluate_model(run_result, eval_type='test'):
    """Evaluate a trained model on val or test data using minimal_evaluate.py"""
    
    print(f"\n{'='*80}")
    print(f"=== Evaluating Run {run_result['run']} on {eval_type.upper()} Data ===")
    
    checkpoint_dir = run_result['checkpoint_dir']
    output_dir = run_result['output_dir']
    
    # Check if checkpoint exists
    checkpoint_path = Path(checkpoint_dir) / 'best_checkpoint'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    # Create output directory for evaluation results
    eval_output_dir = Path(output_dir) / f"{eval_type}_evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data directory from environment or use default
    data_dir = os.environ.get('DATA_DIR', '/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final')
    
    # CSV path
    csv_path = 'data/t15_copyTaskData_description.csv'
    
    # Build command to call minimal_evaluate.py
    cmd = [
        'python',
        'scripts/minimal_evaluate.py',
        '--model_path', str(checkpoint_dir),
        '--eval_type', eval_type,
        '--data_dir', data_dir,
        '--csv_path', csv_path,
        '--output_dir', str(eval_output_dir),
        '--gpu_number', '1',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Parse results from output
        # Look for the summary line with PER
        for line in result.stdout.split('\n'):
            if 'Aggregate Phoneme Error Rate (PER):' in line:
                per = float(line.split(':')[1].strip())
                print(f"[Run {run_result['run']}] {eval_type.upper()} PER: {per:.6f}")
                return {
                    'run': run_result['run'],
                    'eval_type': eval_type,
                    'per': per,
                    'output_dir': str(eval_output_dir)
                }
        
        print(f"Warning: Could not parse PER from evaluation output")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def main():
    # Path to config file
    args_path = "scripts/minimal_baseline_args.yaml"
    
    # Load base config
    base_args = OmegaConf.load(args_path)
    
    original_output_dir = base_args.output_dir
    original_checkpoint_dir = base_args.checkpoint_dir
    
    print(f"\n{'='*80}")
    print(f"Final Model Training - {NUM_RUNS} Runs")
    print(f"Base output dir: {original_output_dir}")
    print(f"Base checkpoint dir: {original_checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Run training multiple times
    training_results = []
    for run_idx in range(NUM_RUNS):
        result = run_single_repeat(run_idx, base_args, original_output_dir, original_checkpoint_dir)
        training_results.append(result)
    
    print(f"\n{'='*80}")
    print(f"=== All Training Complete - Starting Evaluation ===")
    
    # Evaluate all runs on both val and test data
    evaluation_results = []
    for result in training_results:
        # Evaluate on validation set
        val_result = evaluate_model(result, eval_type='val')
        
        # Evaluate on test set
        test_result = evaluate_model(result, eval_type='test')
        
        if val_result and test_result:
            evaluation_results.append({
                'run': result['run'],
                'seed': result['seed'],
                'train_val_per': result['best_val_per'],  # PER from training validation
                'eval_val_per': val_result['per'],         # PER from separate eval on val
                'test_per': test_result['per'],            # PER from eval on test
                'val_eval_dir': val_result['output_dir'],
                'test_eval_dir': test_result['output_dir'],
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"=== FINAL SUMMARY ===")
    
    print("\nTraining Results (sorted by Training Val PER):")
    training_sorted = sorted(training_results, key=lambda x: x["best_val_per"])
    for r in training_sorted:
        print(f"  Run {r['run']} (seed={r['seed']}): Train Val PER={r['best_val_per']:.6f}")
    
    print("\nEvaluation Results (sorted by Test PER):")
    if evaluation_results:
        eval_sorted = sorted(evaluation_results, key=lambda x: x["test_per"])
        
        print("\n  Run | Seed | Train Val PER | Eval Val PER | Test PER")
        print("  " + "-"*65)
        for r in eval_sorted:
            print(f"  {r['run']:3d} | {r['seed']:4d} | {r['train_val_per']:13.6f} | {r['eval_val_per']:12.6f} | {r['test_per']:8.6f}")
        
        # Calculate average and std
        test_pers = [r['test_per'] for r in evaluation_results]
        val_pers = [r['eval_val_per'] for r in evaluation_results]
        
        avg_val_per = np.mean(val_pers)
        std_val_per = np.std(val_pers)
        avg_test_per = np.mean(test_pers)
        std_test_per = np.std(test_pers)
        
        print(f"\nAggregate Performance:")
        print(f"  Average Val PER:  {avg_val_per:.6f} ± {std_val_per:.6f}")
        print(f"  Average Test PER: {avg_test_per:.6f} ± {std_test_per:.6f}")
        print(f"  Best Val PER:     {min(val_pers):.6f}")
        print(f"  Best Test PER:    {min(test_pers):.6f}")
        
        best_run = eval_sorted[0]
        print(f"\nBest Model (by Test PER): Run {best_run['run']}")
        print(f"  Seed: {best_run['seed']}")
        print(f"  Train Val PER: {best_run['train_val_per']:.6f}")
        print(f"  Eval Val PER:  {best_run['eval_val_per']:.6f}")
        print(f"  Test PER:      {best_run['test_per']:.6f}")
    
    print(f"\n{'='*80}")
    print("All runs completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

