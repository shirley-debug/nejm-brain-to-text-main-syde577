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

# --- SGD3: Additional optimizations for SGD training ---
print(f"\n=== SGD3 Configuration (using YAML values + SGD optimizations) ===")
print(f"LR from YAML: {args.lr_max} (main), {args.lr_max_day} (day)")
print(f"LR min from YAML: {args.lr_min} (main), {args.lr_min_day} (day)")
print(f"Weight decay from YAML: {args.weight_decay} (main), {args.weight_decay_day} (day)")

# Use YAML values as-is (already optimized: lr_max=0.015, weight_decay=0.005)
# Additional SGD-specific adjustments:
# - Increase momentum for smoother training (reduces oscillation)
# - Optionally add weight decay to day params if still overfitting

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

# Create param groups with same structure as AdamW setup
# Using YAML values directly (lr_max=0.015, weight_decay=0.005)
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

# Create SGD optimizer with increased momentum for smoother training
trainer.optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.88,  
    nesterov=True,  
)

# Recreate the learning rate scheduler for the new optimizer
if args.lr_scheduler_type == 'cosine':
    trainer.learning_rate_scheduler = trainer.create_cosine_lr_scheduler(trainer.optimizer)
elif args.lr_scheduler_type == 'linear':
    trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=trainer.optimizer,
        start_factor=1.0,
        end_factor=args.lr_min / args.lr_max,
        total_iters=args.lr_decay_steps,
    )

# Train the model
metrics = trainer.train()

# Save metrics to file for later analysis/plotting
# Note: val_metrics.pkl (in checkpoint_dir) contains detailed metrics from the BEST validation step
# This training_metrics.pkl contains the FULL training history for plotting
metrics_file = Path(args.output_dir) / 'training_metrics.pkl'
with open(metrics_file, 'wb') as f:
    pickle.dump({
        'metrics': metrics,
        'args': args,
        'optimizer_info': {
            'type': 'SGD',
            'momentum': 0.88,  
            'nesterov': True,
            'lr_max': args.lr_max,  # From YAML: 0.015
            'lr_max_day': args.lr_max_day,  # From YAML: 0.015
            'lr_min': args.lr_min,  # From YAML: 0.0002
            'lr_min_day': args.lr_min_day,  # From YAML: 0.0002
            'weight_decay': args.weight_decay,  # From YAML: 0.005
            'weight_decay_day': args.weight_decay_day  # From YAML: 0
        }
    }, f)

print(f"\n✓ Saved training metrics to: {metrics_file}")