from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../model_training'))

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import argparse
import pickle

parser = argparse.ArgumentParser(description="Train Brain-to-Text Decoder Model")
parser.add_argument('--config', type=str, default='rnn_args.yaml', help='Path to the config file')
args_cli = parser.parse_args()

args = OmegaConf.load(args_cli.config)
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()

# Save metrics to file for later analysis/plotting
# Note: val_metrics.pkl (in checkpoint_dir) contains detailed metrics from the BEST validation step
# This training_metrics.pkl contains the FULL training history for plotting
metrics_file = Path(args.output_dir) / 'training_metrics.pkl'
with open(metrics_file, 'wb') as f:
    pickle.dump({
        'metrics': metrics,
        'args': args,
    }, f)

print(f"\n✓ Saved training metrics to: {metrics_file}")