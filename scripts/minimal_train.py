import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_training'))

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import argparse
import torch

parser = argparse.ArgumentParser(description="Train Brain-to-Text Decoder Model")
parser.add_argument('--config', type=str, default='rnn_args.yaml', help='Path to the config file')
args_cli = parser.parse_args()

args = OmegaConf.load(args_cli.config)
trainer = BrainToTextDecoder_Trainer(args)

# Override scheduler if step scheduler is requested
if args.get('lr_scheduler_type') == 'step':
    trainer.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=trainer.optimizer,
        step_size=args.get('lr_step_size', 2000),
        gamma=args.get('lr_gamma', 0.3),
    )
    print(f"Using StepLR scheduler: step_size={args.get('lr_step_size', 2000)}, gamma={args.get('lr_gamma', 0.3)}")

metrics = trainer.train()