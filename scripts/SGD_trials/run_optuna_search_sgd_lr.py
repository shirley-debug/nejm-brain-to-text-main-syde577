import optuna
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import os
import numpy as np
import torch 
import shutil
import logging

os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

def objective(trial):
    
    args_path = 'minimal_baseline_args.yaml' 
    if not os.path.exists(args_path):
        print(f"FATAL ERROR: '{args_path}' not found. This script depends on it.")
        return 1e9 
        
    args = OmegaConf.load(args_path)

    base_output_dir = args.output_dir
    base_checkpoint_dir = args.checkpoint_dir
    trial_id_str = f"trial_{trial.number}"
    
    args.output_dir = os.path.join(base_output_dir, trial_id_str)
    args.checkpoint_dir = os.path.join(base_checkpoint_dir, trial_id_str)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
         shutil.rmtree(args.checkpoint_dir)

    # Force Optimizer to SGD
    optimizer_name = "SGD"
    lr = trial.suggest_float("lr", 0.005, 0.1, log=True)
    
    nesterov = trial.suggest_categorical("nesterov", [True, False])
    
    # Optionally tune weight_decay for other_params (or use default from YAML)
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    weight_decay = args.get('weight_decay', 0.001)

    # Overwrite the default args with the new values
    args.lr_max = lr 
    
    # NOTE: args.lr_max_day stays at 0.005 (from YAML)
    # NOTE: args.weight_decay_day stays at 0 (from YAML) - day params typically don't use weight decay
    
    # Train for 2000 batches
    args.num_training_batches = 2000 
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: optimizer={optimizer_name}, nesterov={nesterov}")
    print(f"  Main LR: {lr:.6f} (from trial)")
    print(f"  Day LR:  {args.lr_max_day:.6f} (from YAML)")
    print(f"  Weight decay (other_params): {weight_decay:.6f}")
    print(f"  Weight decay (day_params): {args.weight_decay_day:.6f} (from YAML)")
    print(f"Training for {args.num_training_batches} batches.")
    print(f"Output dir: {args.output_dir}")

    try:
        # Create trainer (initializes model)
        trainer = BrainToTextDecoder_Trainer(args)

        # Manually Create Optimizer and Scheduler
        # Filter for requires_grad=True to only include trainable parameters
        bias_params = [p for name, p in trainer.model.named_parameters() 
                      if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
        day_params = [p for name, p in trainer.model.named_parameters() 
                     if 'day_' in name and p.requires_grad]
        other_params = [p for name, p in trainer.model.named_parameters() 
                       if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name 
                       and p.requires_grad]

        if len(day_params) > 0: 
            param_groups = [
                # Bias params: no weight decay, use main LR
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': lr}, 
                # Day params: use separate LR and weight_decay from YAML (typically 0)
                {'params': day_params, 'lr': args.lr_max_day, 'weight_decay': args.weight_decay_day, 
                 'group_type': 'day_layer'},
                # Other params: use main LR and weight_decay (from YAML or tuned)
                {'params': other_params, 'group_type': 'other', 'lr': lr, 'weight_decay': weight_decay} 
            ]
        else: 
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': lr}, 
                {'params': other_params, 'group_type': 'other', 'lr': lr, 'weight_decay': weight_decay} 
            ]

        # Force SGD
        print("Initializing SGD optimizer...")
        trainer.optimizer = torch.optim.SGD(
            param_groups, 
            momentum=0.9,  # Standard momentum value, could also be tuned
            nesterov=nesterov,  # Pass the boolean here
            # lr is NOT passed here (it is in param_groups)
            # weight_decay is NOT passed here (it is in param_groups)
        ) 

        # Recreate the scheduler for the new optimizer
        if args.lr_scheduler_type == 'cosine':
            trainer.learning_rate_scheduler = trainer.create_cosine_lr_scheduler(trainer.optimizer)
        elif args.lr_scheduler_type == 'linear':
            trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=trainer.optimizer, 
                start_factor=1.0,
                end_factor=args.lr_min / args.lr_max, 
                total_iters=args.lr_decay_steps,
            )
        else:
            trainer.learning_rate_scheduler = None

        # Run Training
        train_stats = trainer.train() 

        val_per_list = train_stats.get('val_PERs', []) 
        val_score = np.min(val_per_list) if val_per_list else 1.0 
        
        print(f"Trial {trial.number} finished. Best (min) PER: {val_score:.4f}")

        return val_score
    
    except Exception as e:
        print(f"!!! TRIAL {trial.number} FAILED with exception: {e}")
        logging.exception(f"Trial {trial.number} failed.")
        return 1.0 

if __name__ == "__main__":
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) 

    print("\n--- Optimization Finished ---")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Min PER): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

