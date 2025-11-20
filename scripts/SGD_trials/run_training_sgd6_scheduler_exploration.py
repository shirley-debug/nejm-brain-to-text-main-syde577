"""
SGD6: Scheduler Exploration
Based on SGD5 (best Optuna trial: momentum=0.88, weight_decay=0.004464)
Tests different learning rate schedulers to find optimal schedule.
"""

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import os
import numpy as np
import torch 
import shutil
import pickle
import math
from pathlib import Path

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

args_path = 'minimal_sgd_args.yaml'
print(f"Loading configuration from: {args_path}")
args = OmegaConf.load(args_path)

# --- SGD6: Scheduler Exploration ---
# Using best hyperparameters from Optuna (Trial 5)
# momentum=0.88, weight_decay=0.004464
print(f"\n{'='*80}")
print(f"SGD6: Scheduler Exploration - Testing All Schedulers")
print(f"{'='*80}")
print(f"Base config: momentum=0.88, weight_decay=0.004464 (from Optuna best trial)")
print(f"LR from YAML: {args.lr_max} (main), {args.lr_max_day} (day)")

SCHEDULERS_TO_TEST = [
    'cosine',
    'step_gentle',
    'step_moderate',
    'step_aggressive',
    'exponential',
    'polynomial',
    'onecycle',
]

print(f"\nWill test {len(SCHEDULERS_TO_TEST)} schedulers: {SCHEDULERS_TO_TEST}")

# Store original paths
original_output_dir = args.output_dir
original_checkpoint_dir = args.checkpoint_dir

results_summary = []

for scheduler_idx, SCHEDULER_TYPE in enumerate(SCHEDULERS_TO_TEST, 1):
    print(f"\n{'#'*80}")
    print(f"# SCHEDULER {scheduler_idx}/{len(SCHEDULERS_TO_TEST)}: {SCHEDULER_TYPE.upper()}")
    print(f"{'#'*80}\n")
    
    # Create unique directories for this scheduler
    args.output_dir = f"{original_output_dir}_{SCHEDULER_TYPE}"
    args.checkpoint_dir = f"{original_checkpoint_dir}_{SCHEDULER_TYPE}"
    
    # Clean up existing directories
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)  
    if os.path.exists(args.checkpoint_dir):
        print(f"Removing existing checkpoint directory: {args.checkpoint_dir}")
        shutil.rmtree(args.checkpoint_dir)
    
    # Create trainer with updated paths
    trainer = BrainToTextDecoder_Trainer(args)
    
    # --- Replace AdamW optimizer with SGD (using Optuna best params) ---
    print(f"\n[{SCHEDULER_TYPE}] Replacing AdamW optimizer with SGD (Optuna best config)...")
    
    # Get parameter groups (same structure as original)
    bias_params = [p for name, p in trainer.model.named_parameters() 
                   if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
    day_params = [p for name, p in trainer.model.named_parameters() 
                  if 'day_' in name and p.requires_grad]
    other_params = [p for name, p in trainer.model.named_parameters() 
                    if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name 
                    and p.requires_grad]
    
    # Create param groups with Optuna best weight decay
    if len(day_params) > 0:
        param_groups = [
            {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max},
            {'params': day_params, 'lr': args.lr_max_day, 'weight_decay': args.weight_decay_day, 
             'group_type': 'day_layer'},
            {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
             'weight_decay': 0.004464}  # Optuna best weight decay
        ]
    else:
        param_groups = [
            {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max},
            {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
             'weight_decay': 0.004464}  # Optuna best weight decay
        ]
    
    # Create SGD optimizer with Optuna best momentum
    trainer.optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.88,  # Optuna best momentum
        nesterov=True,
    )
    
    lr_warmup_steps = args.get('lr_warmup_steps', 1000)
    lr_warmup_steps_day = args.get('lr_warmup_steps_day', 1000)
    num_training_batches = args.num_training_batches
    
    def create_lr_lambdas(lambda_func):
        """Helper to create lambda list for all parameter groups."""
        num_groups = len(trainer.optimizer.param_groups)
        if num_groups == 3:
            return [
                lambda step: lambda_func(step, lr_warmup_steps),  # biases
                lambda step: lambda_func(step, lr_warmup_steps_day),  # day params
                lambda step: lambda_func(step, lr_warmup_steps),  # other params
            ]
        elif num_groups == 2:
            return [
                lambda step: lambda_func(step, lr_warmup_steps),  # biases
                lambda step: lambda_func(step, lr_warmup_steps),  # other params
            ]
        else:
            raise ValueError(f"Unexpected number of param groups: {num_groups}")
    
    print(f"\n[{SCHEDULER_TYPE}] Creating {SCHEDULER_TYPE} scheduler...")
    
    if SCHEDULER_TYPE == 'cosine':
        # Cosine annealing (baseline - same as SGD3/SGD5)
        print(f"[{SCHEDULER_TYPE}] Using cosine annealing scheduler (baseline)")
        trainer.learning_rate_scheduler = trainer.create_cosine_lr_scheduler(trainer.optimizer)
        scheduler_info = {
            'type': 'cosine',
            'lr_max': args.lr_max,
            'lr_min': args.lr_min,
            'warmup_steps': lr_warmup_steps,
            'decay_steps': args.lr_decay_steps
        }
    
    elif SCHEDULER_TYPE == 'step_gentle':
        # Gentle step decay: smaller gamma, less frequent steps
        # Milestones at 1/3, 2/3 of training (3333, 6666)
        # Gamma = 0.5 (reduce by half each time)
        milestones = [3333, 6666]
        gamma = 0.5
        
        def step_decay_with_warmup_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            decay_count = sum(1 for m in milestones if step >= m)
            return gamma ** decay_count
        
        lr_lambdas = create_lr_lambdas(step_decay_with_warmup_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'step_gentle',
            'milestones': milestones,
            'gamma': gamma,
            'warmup_steps': lr_warmup_steps
        }
        print(f"[{SCHEDULER_TYPE}] Milestones: {milestones}")
        print(f"[{SCHEDULER_TYPE}] Gamma: {gamma} (reduce by half each step)")
    
    elif SCHEDULER_TYPE == 'step_moderate':
        # Moderate step decay: more frequent steps, moderate gamma
        # Milestones every 2000 steps (2000, 4000, 6000, 8000)
        # Gamma = 0.7 (reduce by 30% each time)
        milestones = [2000, 4000, 6000, 8000]
        gamma = 0.7
        
        def step_decay_with_warmup_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            decay_count = sum(1 for m in milestones if step >= m)
            return gamma ** decay_count
        
        lr_lambdas = create_lr_lambdas(step_decay_with_warmup_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'step_moderate',
            'milestones': milestones,
            'gamma': gamma,
            'warmup_steps': lr_warmup_steps
        }
        print(f"[{SCHEDULER_TYPE}] Milestones: {milestones}")
        print(f"[{SCHEDULER_TYPE}] Gamma: {gamma} (reduce by 30% each step)")
    
    elif SCHEDULER_TYPE == 'step_aggressive':
        # Aggressive step decay: similar to SGD4 but with better gamma
        # Milestones every 2500 steps (2500, 5000, 7500)
        # Gamma = 0.3 (reduce by 70% each time - less aggressive than SGD4's 0.1)
        milestones = [2500, 5000, 7500]
        gamma = 0.3
        
        def step_decay_with_warmup_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            decay_count = sum(1 for m in milestones if step >= m)
            return gamma ** decay_count
        
        lr_lambdas = create_lr_lambdas(step_decay_with_warmup_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'step_aggressive',
            'milestones': milestones,
            'gamma': gamma,
            'warmup_steps': lr_warmup_steps
        }
        print(f"[{SCHEDULER_TYPE}] Milestones: {milestones}")
        print(f"[{SCHEDULER_TYPE}] Gamma: {gamma} (reduce by 70% each step)")
    
    elif SCHEDULER_TYPE == 'exponential':
        # Exponential decay: smooth exponential curve
        # Decay rate calculated to reach lr_min at lr_decay_steps
        decay_rate = (args.lr_min / args.lr_max) ** (1.0 / (args.lr_decay_steps - lr_warmup_steps))
        
        def exponential_decay_with_warmup_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            steps_since_warmup = step - warmup_steps
            return decay_rate ** steps_since_warmup
        
        lr_lambdas = create_lr_lambdas(exponential_decay_with_warmup_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'exponential',
            'decay_rate': decay_rate,
            'warmup_steps': lr_warmup_steps,
            'decay_steps': args.lr_decay_steps
        }
        print(f"[{SCHEDULER_TYPE}] Decay rate: {decay_rate:.6f}")
        print(f"[{SCHEDULER_TYPE}] Will reach lr_min={args.lr_min} at step {args.lr_decay_steps}")
    
    elif SCHEDULER_TYPE == 'polynomial':
        # Polynomial decay: (1 - progress)^power
        # Power = 2 (quadratic decay)
        power = 2.0
        
        def polynomial_decay_with_warmup_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if step >= args.lr_decay_steps:
                return args.lr_min / args.lr_max
            progress = float(step - warmup_steps) / float(max(1, args.lr_decay_steps - warmup_steps))
            min_lr_ratio = args.lr_min / args.lr_max
            return min_lr_ratio + (1 - min_lr_ratio) * ((1 - progress) ** power)
        
        lr_lambdas = create_lr_lambdas(polynomial_decay_with_warmup_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'polynomial',
            'power': power,
            'warmup_steps': lr_warmup_steps,
            'decay_steps': args.lr_decay_steps
        }
        print(f"[{SCHEDULER_TYPE}] Power: {power} (quadratic decay)")
    
    elif SCHEDULER_TYPE == 'onecycle':
        # OneCycleLR: triangular policy with max at 1/3 of training
        # Note: PyTorch's OneCycleLR doesn't support warmup easily, so we approximate
        max_lr = args.lr_max
        pct_start = 0.33  # Peak at 33% of training
        
        # Approximate OneCycle with custom lambda
        def onecycle_lambda(step, warmup_steps):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            total_steps = args.lr_decay_steps - warmup_steps
            step_in_cycle = step - warmup_steps
            if step_in_cycle < total_steps * pct_start:
                # Increasing phase
                progress = step_in_cycle / (total_steps * pct_start)
                return 0.1 + 0.9 * progress  # From 10% to 100%
            else:
                # Decreasing phase
                progress = (step_in_cycle - total_steps * pct_start) / (total_steps * (1 - pct_start))
                min_lr_ratio = args.lr_min / args.lr_max
                return max(min_lr_ratio, 1.0 - 0.9 * progress)  # From 100% to min_lr_ratio
        
        lr_lambdas = create_lr_lambdas(onecycle_lambda)
        trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=trainer.optimizer,
            lr_lambda=lr_lambdas
        )
        scheduler_info = {
            'type': 'onecycle',
            'pct_start': pct_start,
            'warmup_steps': lr_warmup_steps
        }
        print(f"[{SCHEDULER_TYPE}] Peak at: {pct_start*100}% of training")
    
    else:
        raise ValueError(f"Unknown scheduler type: {SCHEDULER_TYPE}")
    
    print(f"\n[{SCHEDULER_TYPE}] SGD6 Optimizer Configuration:")
    print(f"  Main LR: {args.lr_max} (from YAML)")
    print(f"  Day LR: {args.lr_max_day} (from YAML)")
    print(f"  Momentum: 0.88 (Optuna best)")
    print(f"  Nesterov: True")
    print(f"  Weight decay (other_params): 0.004464 (Optuna best)")
    print(f"  Weight decay (day_params): {args.weight_decay_day} (from YAML)")
    print(f"  LR Scheduler: {SCHEDULER_TYPE}")
    print(f"  Early stopping: {args.early_stopping} (patience: {args.early_stopping_val_steps})")
    
    # Train the model
    print(f"\n[{SCHEDULER_TYPE}] Starting training...")
    try:
        metrics = trainer.train()
        
        # Extract best PER for summary
        val_per_list = metrics.get('val_PERs', [])
        best_per = min(val_per_list) if val_per_list else float('inf')
        
        # Save metrics to file for later analysis/plotting
        # Save with scheduler name in filename for easy identification
        metrics_file = Path(args.output_dir) / f'training_metrics_{SCHEDULER_TYPE}.pkl'
        with open(metrics_file, 'wb') as f:
            pickle.dump({
                'metrics': metrics,
                'args': args,
                'optimizer_info': {
                    'type': 'SGD',
                    'momentum': 0.88,  # Optuna best
                    'nesterov': True,
                    'lr_max': args.lr_max,
                    'lr_max_day': args.lr_max_day,
                    'weight_decay': 0.004464,  # Optuna best
                    'weight_decay_day': args.weight_decay_day,
                    'scheduler_info': scheduler_info,
                    'scheduler_type': SCHEDULER_TYPE  # Add scheduler type for easy identification
                }
            }, f)
        
        # Also save a copy in the parent directory for easy access
        parent_metrics_file = Path(original_output_dir).parent / f'training_metrics_{SCHEDULER_TYPE}.pkl'
        with open(parent_metrics_file, 'wb') as f:
            pickle.dump({
                'metrics': metrics,
                'args': args,
                'optimizer_info': {
                    'type': 'SGD',
                    'momentum': 0.88,  # Optuna best
                    'nesterov': True,
                    'lr_max': args.lr_max,
                    'lr_max_day': args.lr_max_day,
                    'weight_decay': 0.004464,  # Optuna best
                    'weight_decay_day': args.weight_decay_day,
                    'scheduler_info': scheduler_info,
                    'scheduler_type': SCHEDULER_TYPE
                }
            }, f)
        
        print(f"\n[{SCHEDULER_TYPE}] ✓ Training completed!")
        print(f"[{SCHEDULER_TYPE}] ✓ Best PER: {best_per:.6f}")
        print(f"[{SCHEDULER_TYPE}] ✓ Metrics saved to: {metrics_file}")
        print(f"[{SCHEDULER_TYPE}] ✓ Also saved to: {parent_metrics_file} (for easy plotting)")
        
        # Store result for summary
        results_summary.append({
            'scheduler': SCHEDULER_TYPE,
            'best_per': best_per,
            'output_dir': args.output_dir,
            'metrics_file': str(metrics_file),
            'parent_metrics_file': str(parent_metrics_file)
        })
        
    except Exception as e:
        print(f"\n[{SCHEDULER_TYPE}] ✗ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        results_summary.append({
            'scheduler': SCHEDULER_TYPE,
            'best_per': float('inf'),
            'error': str(e)
        })
    
    print(f"\n[{SCHEDULER_TYPE}] {'='*60}\n")

# Sort by best PER
results_summary_sorted = sorted(results_summary, key=lambda x: x['best_per'])

print(f"{'Rank':<6} {'Scheduler':<20} {'Best PER':<12} {'Status':<20}")
print(f"{'-'*80}")
for rank, result in enumerate(results_summary_sorted, 1):
    scheduler = result['scheduler']
    best_per = result['best_per']
    if 'error' in result:
        status = f"ERROR: {result['error'][:30]}"
    else:
        status = "✓ Success"
    print(f"{rank:<6} {scheduler:<20} {best_per:<12.6f} {status:<20}")

print(f"\n{'='*80}")
print(f"Best scheduler: {results_summary_sorted[0]['scheduler']} (PER: {results_summary_sorted[0]['best_per']:.6f})")
print(f"{'='*80}\n")

# Save summary to file
summary_file = Path(original_output_dir).parent / 'scheduler_experiment_summary.pkl'
with open(summary_file, 'wb') as f:
    pickle.dump({
        'results': results_summary_sorted,
        'best_scheduler': results_summary_sorted[0]['scheduler'],
        'best_per': results_summary_sorted[0]['best_per'],
        'metrics_files': {r['scheduler']: r.get('parent_metrics_file', '') for r in results_summary_sorted if 'parent_metrics_file' in r}
    }, f)
print(f"Summary saved to: {summary_file}")

