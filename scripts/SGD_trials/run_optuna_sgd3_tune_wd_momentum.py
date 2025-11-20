"""
Optuna hyperparameter optimization for SGD3.
Tunes weight_decay and momentum while keeping cosine LR decay from SGD3.
"""
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
    
    # Load the SGD config (same as SGD3)
    args_path = 'minimal_sgd_args.yaml' 
    if not os.path.exists(args_path):
        print(f"FATAL ERROR: '{args_path}' not found. This script depends on it.")
        return 1e9 
        
    args = OmegaConf.load(args_path)

    # Give each trial a unique output and checkpoint directory
    base_output_dir = args.output_dir
    base_checkpoint_dir = args.checkpoint_dir
    trial_id_str = f"trial_{trial.number}"
    
    args.output_dir = os.path.join(base_output_dir, trial_id_str)
    args.checkpoint_dir = os.path.join(base_checkpoint_dir, trial_id_str)

    # Clean up old directories
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
         shutil.rmtree(args.checkpoint_dir)

    # Tune Hyperparameters
    # Tune momentum: 0.85 to 0.97
    momentum = trial.suggest_float("momentum", 0.85, 0.97, step=0.01)
    
    # Tune weight_decay: reasonable range (0.001 to 0.02)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.02, log=True)
    
    # Keep everything else from SGD3:
    # - LR from YAML (0.015)
    # - Cosine LR decay (from YAML)
    # - Nesterov: True
    # - weight_decay_day: 0 (from YAML)
    
    # Overwrite weight_decay in args (for param groups)
    args.weight_decay = weight_decay
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Momentum: {momentum:.3f} (tuned)")
    print(f"  Weight decay: {weight_decay:.6f} (tuned)")
    print(f"  Main LR: {args.lr_max:.6f} (from YAML)")
    print(f"  Day LR: {args.lr_max_day:.6f} (from YAML)")
    print(f"  Weight decay (day_params): {args.weight_decay_day:.6f} (from YAML)")
    print(f"  LR Scheduler: {args.lr_scheduler_type} (from YAML, same as SGD3)")
    print(f"  Training for {args.num_training_batches} batches")
    print(f"  Output dir: {args.output_dir}")

    try:
        # Create trainer (initializes model)
        trainer = BrainToTextDecoder_Trainer(args)

        # Create Optimizer with Tuned Parameters
        # Filter for requires_grad=True
        bias_params = [p for name, p in trainer.model.named_parameters() 
                      if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
        day_params = [p for name, p in trainer.model.named_parameters() 
                     if 'day_' in name and p.requires_grad]
        other_params = [p for name, p in trainer.model.named_parameters() 
                       if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name 
                       and p.requires_grad]

        # Create param groups with tuned weight_decay
        if len(day_params) > 0: 
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max}, 
                {'params': day_params, 'lr': args.lr_max_day, 'weight_decay': args.weight_decay_day, 
                 'group_type': 'day_layer'},
                {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
                 'weight_decay': weight_decay}  # Use tuned weight_decay
            ]
        else: 
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': args.lr_max}, 
                {'params': other_params, 'group_type': 'other', 'lr': args.lr_max, 
                 'weight_decay': weight_decay}  # Use tuned weight_decay
            ]

        # Create SGD optimizer with tuned momentum
        print("Initializing SGD optimizer with tuned parameters...")
        trainer.optimizer = torch.optim.SGD(
            param_groups, 
            momentum=momentum,  # Tuned momentum
            nesterov=True,  # Keep Nesterov as in SGD3
        ) 

        # Create Cosine LR scheduler (same as SGD3)
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

        # Get best validation PER
        val_per_list = train_stats.get('val_PERs', []) 
        val_score = np.min(val_per_list) if val_per_list else 1.0 
        
        print(f"Trial {trial.number} finished. Best (min) PER: {val_score:.4f}")

        return val_score
    
    except Exception as e:
        print(f"!!! TRIAL {trial.number} FAILED with exception: {e}")
        logging.exception(f"Trial {trial.number} failed.")
        return 1.0 

if __name__ == "__main__":
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',  # Minimize PER
        study_name='sgd3_wd_momentum_tuning'
    )
    
    # Run optimization
    print("Starting Optuna optimization for SGD3 (weight_decay + momentum)...")
    print("Search space:")
    print("  - Momentum: 0.85 to 0.97 (step=0.01)")
    print("  - Weight decay: 0.001 to 0.02 (log scale)")
    print("  - Cosine LR decay: from YAML (fixed)")
    print("  - LR: 0.015 (from YAML, fixed)")
    print("  - Nesterov: True (fixed)")
    print()
    
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed
    
    # Print best trial
    trial = study.best_trial
    print(f"\nBest trial:")
    print(f"  Value (Min PER): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Print top 3 trials
    print(f"\nTop 3 trials:")
    for i, trial in enumerate(study.trials_dataframe().nsmallest(3, 'value').itertuples()):
        print(f"  {i+1}. PER: {trial.value:.4f}, momentum: {trial.momentum:.3f}, weight_decay: {trial.weight_decay:.6f}")
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

