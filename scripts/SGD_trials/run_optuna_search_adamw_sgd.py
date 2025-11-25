import optuna
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import os
import numpy as np
import torch 
import shutil
import logging
-
# This MUST be set before loading args so ${oc.env:DATA_DIR} resolves
os.environ['DATA_DIR'] = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"

def objective(trial):
    
    args_path = 'minimal_baseline_args.yaml' 
    if not os.path.exists(args_path):
        print(f"FATAL ERROR: '{args_path}' not found. This script depends on it.")
        return 1e9 
        
    args = OmegaConf.load(args_path)

    # Give each trial a unique output and checkpoint directory
    # Use paths from the loaded config as a base
    base_output_dir = args.output_dir
    base_checkpoint_dir = args.checkpoint_dir
    trial_id_str = f"trial_{trial.number}"
    
    args.output_dir = os.path.join(base_output_dir, trial_id_str)
    args.checkpoint_dir = os.path.join(base_checkpoint_dir, trial_id_str)

    # Clean up old directories for this specific trial
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    if os.path.exists(args.checkpoint_dir):
         print(f"Removing existing checkpoint directory: {args.checkpoint_dir}")
         shutil.rmtree(args.checkpoint_dir)

    # Suggest new hyperparameters for this trial
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Overwrite the default args with the new values from Optuna
    # This lr will be used for the 'main' model parts
    args.lr_max = lr 
    # We will leave args.lr_max_day as specified in the YAML

    # Override to train faster for the optimization search
    # The YAML has batches_per_val_step: 2000. 
    # Setting this to 2000 means we train for 2000 steps and run validation *once*.
    # This is efficient for a hyperparameter search.
    args.num_training_batches = 2000 
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: optimizer={optimizer_name}, lr={lr:.6f}")
    print(f"  Main LR: {lr:.6f} (from trial)")
    print(f"  Day LR: {args.lr_max_day:.6f} (from YAML)")
    print(f"  Weight decay (other_params): {args.weight_decay:.6f} (from YAML)")
    print(f"  Weight decay (day_params): {args.weight_decay_day:.6f} (from YAML)")
    print(f"Training for {args.num_training_batches} batches.")
    print(f"Output dir: {args.output_dir}")

    try:
        # Create trainer (initializes model)
        trainer = BrainToTextDecoder_Trainer(args)

        # This logic overwrites the default optimizer created by the trainer
        
        bias_params = [p for name, p in trainer.model.named_parameters() if ('gru.bias' in name or 'out.bias' in name) and p.requires_grad]
        day_params = [p for name, p in trainer.model.named_parameters() if 'day_' in name and p.requires_grad]
        other_params = [p for name, p in trainer.model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name and p.requires_grad]

        # Match the structure from rnn_trainer.py create_optimizer()
        if len(day_params) > 0: 
            print(f"Found {len(day_params)} day-specific parameters.")
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': lr},
                {'params': day_params, 'lr': args.lr_max_day, 'weight_decay': args.weight_decay_day, 'group_type': 'day_layer'},
                {'params': other_params, 'group_type': 'other', 'lr': lr, 'weight_decay': args.weight_decay}
            ]
        else: 
            print("No day-specific parameters found (or 'day_' not in name).")
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias', 'lr': lr}, 
                {'params': other_params, 'group_type': 'other', 'lr': lr, 'weight_decay': args.weight_decay}
            ]

        # Now create the optimizer using the filtered groups
        if optimizer_name == "AdamW":
            print("Using AdamW optimizer.")
            trainer.optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.lr_max,  # Default LR (though groups override this)
                betas=(args.beta0, args.beta1),
                eps=args.epsilon,
                weight_decay=args.weight_decay,  # Default weight_decay (though groups override this)
                fused=True 
            )
                    
        elif optimizer_name == "SGD":
            print("Using SGD optimizer.")
            trainer.optimizer = torch.optim.SGD(
                param_groups, 
                momentum=0.9,  # Standard momentum value
                nesterov=True,  # Use Nesterov accelerated gradient
                # lr and weight_decay are set in param_groups
            ) 

        if args.lr_scheduler_type == 'cosine':
            print("Recreating cosine LR scheduler.")
            trainer.learning_rate_scheduler = trainer.create_cosine_lr_scheduler(trainer.optimizer)
        elif args.lr_scheduler_type == 'linear':
            print("Recreating linear LR scheduler.")
            trainer.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=trainer.optimizer, 
                start_factor=1.0,
                end_factor=args.lr_min / args.lr_max, 
                total_iters=args.lr_decay_steps,
            )
        else:
            print(f"Warning: Unknown lr_scheduler_type '{args.lr_scheduler_type}'. No scheduler will be used.")
            trainer.learning_rate_scheduler = None

        # Run Training
        train_stats = trainer.train() 

        # Get the list of all validation Phoneme Error Rates
        val_per_list = train_stats.get('val_PERs', []) 
        
        # Get the best (minimum) PER from the list.
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