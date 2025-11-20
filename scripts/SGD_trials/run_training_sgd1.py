%from omegaconf import OmegaConf
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

# Uses paths directly from the loaded config
if os.path.exists(args.output_dir):
    print(f"Removing existing output directory: {args.output_dir}")
    shutil.rmtree(args.output_dir)  
if os.path.exists(args.checkpoint_dir):
    print(f"Removing existing checkpoint directory: {args.checkpoint_dir}")
    shutil.rmtree(args.checkpoint_dir)

trainer = BrainToTextDecoder_Trainer(args)

# --- Replace AdamW optimizer with SGD ---
print("Replacing AdamW optimizer with SGD...")

# Get parameter groups (same structure as original)
bias_params = [p for name, p in trainer.model.named_parameters() 
               if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
day_params = [p for name, p in trainer.model.named_parameters() 
              if 'day_' in name and p.requires_grad]
other_params = [p for name, p in trainer.model.named_parameters() 
                if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name 
                and p.requires_grad]

# Create param groups with same structure as AdamW setup
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

# Create SGD optimizer with default parameters
trainer.optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.9,  # Standard momentum value
    nesterov=True,  # Use Nesterov accelerated gradient
    # lr and weight_decay are set in param_groups
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

print(f"SGD optimizer created with:")
print(f"  Main LR: {args.lr_max}")
print(f"  Day LR: {args.lr_max_day}")
print(f"  Momentum: 0.9")
print(f"  Nesterov: True")
print(f"  Weight decay (other_params): {args.weight_decay}")
print(f"  Weight decay (day_params): {args.weight_decay_day}")

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
            'momentum': 0.9,
            'nesterov': True,
            'lr_max': args.lr_max,
            'lr_max_day': args.lr_max_day,
            'weight_decay': args.weight_decay,
            'weight_decay_day': args.weight_decay_day
        }
    }, f)

print(f"\nSaved training metrics to: {metrics_file}")