import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_training'))

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import argparse

parser = argparse.ArgumentParser(description="Train Brain-to-Text Decoder Model")
parser.add_argument('--config', type=str, default='rnn_args.yaml', help='Path to the config file')
args_cli = parser.parse_args()

args = OmegaConf.load(args_cli.config)
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()