from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import os
import numpy as np
import torch 
import shutil
import pickle
from pathlib import Path

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

args_path = 'minimal_sgd_args.yaml'
print(f"Loading configuration from: {args_path}")
args = OmegaConf.load(args_path)

# --- SGD4: Step LR Decay (keep weight decay same as YAML) ---
print(f"\n=== SGD4 Configuration ===")
print(f"LR from YAML: {args.lr_max} (main), {args.lr_max_day} (day)")
print(f"Weight decay from YAML: {args.weight_decay} (main), {args.weight_decay_day} (day)")
print(f"  (Keeping weight decay same as YAML - only changing LR scheduler to step decay)")

# Step LR decay configuration
# Original: decay every 4000 steps with gamma=0.1 (for 120k batches)
# Scaled for 10k batches: 4000 * (10000/120000) = 4000 * 0.083 = 333 steps
# More practical: use 2500 steps per decay (gives ~4 decay steps in 10k batches)
step_decay_interval = 2500  # Decay every 2500 steps
step_decay_gamma = 0.1  # Reduce LR by 10x at each step
num_training_batches = args.num_training_batches

# Calculate milestones for step decay
# With 10k batches and 2500 step interval: milestones at [2500, 5000, 7500]
milestones = list(range(step_decay_interval, num_training_batches, step_decay_interval))
lr_warmup_steps = args.get('lr_warmup_steps', 0)
print(f"\nStep LR Decay Configuration:")
print(f"  Decay interval: {step_decay_interval} steps")
print(f"  Decay factor (gamma): {step_decay_gamma}")
print(f"  Warmup steps: {lr_warmup_steps}")
print(f"  Milestones: {milestones}")
print(f"  Expected LR schedule:")
if lr_warmup_steps > 0:
    print(f"    Steps 0-{lr_warmup_steps}: warmup (0 → {args.lr_max})")
    if milestones and milestones[0] > lr_warmup_steps:
        print(f"    Steps {lr_warmup_steps}-{milestones[0]}: {args.lr_max}")
else:
    if milestones:
        print(f"    Steps 0-{milestones[0]}: {args.lr_max}")
for i, milestone in enumerate(milestones):
    expected_lr = args.lr_max * (step_decay_gamma ** (i + 1))
    next_milestone = milestones[i + 1] if i + 1 < len(milestones) else num_training_batches
    print(f"    Steps {milestone}-{next_milestone}: {expected_lr:.6f}")

print(f"\nWeight Decay Configuration (from YAML, unchanged):")
print(f"  Weight decay (other_params): {args.weight_decay}")
print(f"  Weight decay (day_params): {args.weight_decay_day}")

# Uses paths directly from the loaded config
if os.path.exists(args.output_dir):
    print(f"Removing existing output directory: {args.output_dir}")
    shutil.rmtree(args.output_dir)  
if os.path.exists(args.checkpoint_dir):
    print(f"Removing existing checkpoint directory: {args.checkpoint_dir}")
    shutil.rmtree(args.checkpoint_dir)

trainer = BrainToTextDecoder_Trainer(args)

# --- Replace AdamW optimizer with SGD ---
print("\nReplacing AdamW optimizer with SGD...")

# Get parameter groups (same structure as original)
bias_params = [p for name, p in trainer.model.named_parameters() 
               if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
day_params = [p for name, p in trainer.model.named_parameters() 
              if 'day_' in name and p.requires_grad]
other_params = [p for name, p in trainer.model.named_parameters() 
                if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name 
                and p.requires_grad]

# Create param groups with increased weight decay
if len(day_params) > 0:
    param_groups = [
        {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max},
        {'params': day_params, 'lr': args.lr_max_day, 'weight_decay': args.weight_decay_day, 
         'group_type': 'day_layer'},
        {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
         'weight_decay': args.weight_decay}
    ]
else:
    param_groups = [
        {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max},
        {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
         'weight_decay': args.weight_decay}
    ]

# Create SGD optimizer
trainer.optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.95,
    nesterov=True,
    # lr and weight_decay are set in param_groups
)

# Create Step LR scheduler with warmup support
# Use LambdaLR to combine warmup with step decay
# (lr_warmup_steps already defined above)

def step_decay_with_warmup_lambda(epoch):
    """
    Lambda function for step decay with optional warmup.
    Returns multiplier for learning rate.
    """
    # Warmup phase
    if lr_warmup_steps > 0 and epoch < lr_warmup_steps:
        return float(epoch) / float(max(1, lr_warmup_steps))
    
    # Step decay phase
    # Count how many milestones we've passed
    decay_count = sum(1 for m in milestones if epoch >= m)
    return step_decay_gamma ** decay_count

# Create LambdaLR scheduler that applies the lambda function
trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=trainer.optimizer,
    lr_lambda=step_decay_with_warmup_lambda
)

if lr_warmup_steps > 0:
    print(f"  Warmup steps: {lr_warmup_steps} (LR will ramp from 0 to {args.lr_max})")
else:
    print(f"  No warmup (LR starts at {args.lr_max})")

print(f"\n=== SGD4 Optimizer Configuration ===")
print(f"  Main LR: {args.lr_max} (from YAML)")
print(f"  Day LR: {args.lr_max_day} (from YAML)")
print(f"  Momentum: 0.95")
print(f"  Nesterov: True")
print(f"  Weight decay (other_params): {args.weight_decay} (from YAML)")
print(f"  Weight decay (day_params): {args.weight_decay_day} (from YAML)")
print(f"  LR Scheduler: LambdaLR (step decay with warmup)")
print(f"    - Warmup steps: {lr_warmup_steps}")
print(f"    - Decay milestones: {milestones}")
print(f"    - Decay factor (gamma): {step_decay_gamma}")
print(f"  Early stopping: {args.early_stopping} (patience: {args.early_stopping_val_steps})")

# Train the model
metrics = trainer.train()

# Save metrics to file for later analysis/plotting
metrics_file = Path(args.output_dir) / 'training_metrics.pkl'
with open(metrics_file, 'wb') as f:
    pickle.dump({
        'metrics': metrics,
        'args': args,
        'optimizer_info': {
            'type': 'SGD',
            'momentum': 0.95,
            'nesterov': True,
            'lr_max': args.lr_max,
            'lr_max_day': args.lr_max_day,
            'weight_decay': args.weight_decay,
            'weight_decay_day': args.weight_decay_day,
            'lr_scheduler': 'LambdaLR',
            'lr_scheduler_type': 'step_decay_with_warmup',
            'step_decay_milestones': milestones,
            'step_decay_gamma': step_decay_gamma,
            'step_decay_interval': step_decay_interval,
            'lr_warmup_steps': lr_warmup_steps
        }
    }, f)

print(f"\nSaved training metrics to: {metrics_file}")