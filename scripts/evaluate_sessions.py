import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers import *
import torchaudio.functional as F # for edit distance

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for RNN model inference. Set to -1 to use CPU.')
parser.add_argument('--sessions', type=str, nargs='+', default=None,
                    help='Specify one or more sessions to evaluate (e.g., "t15.2023.08.18" "t15.2023.08.20"). '
                         'If not specified, all sessions in the model config will be evaluated.')
args = parser.parse_args()

# paths to model and data directories
# Note: these paths are relative to the current working directory
model_path = args.model_path
data_dir = args.data_dir

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

assert eval_type != 'train', "You shouldn't be evaluating on the train set."

# load csv file
b2txt_csv_df = pd.read_csv(args.csv_path)

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'args.yaml'))

# set up gpu device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
model = GRUDecoder(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'], 
    n_days = len(model_args['dataset']['sessions']),
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'best_checkpoint'), weights_only=False, map_location=device)
# rename keys to not start with "module." (happens if model was saved with DataParallel)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

# Calculate loss 
# connectionist temporal classification (CTC) → specifically designed for sequence-to-sequence
# tasks where the input and output sequences are not perfectly aligned
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

sessions_to_evaluate = model_args['dataset']['sessions']
if args.sessions:  # If specific sessions were passed
    # Validate that all requested sessions are in the model config
    invalid_sessions = [s for s in args.sessions if s not in sessions_to_evaluate]
    if invalid_sessions:
        raise ValueError(f"Session(s) {invalid_sessions} not found in model config list: {sessions_to_evaluate}")
    
    sessions_to_evaluate = args.sessions  # Overwrite to only evaluate specified sessions
    print(f"Evaluating ONLY specified sessions: {', '.join(args.sessions)}")
else:
    print(f"Evaluating ALL sessions from model config")

# load data for each session
test_data = {}
total_test_trials = 0
for session in sessions_to_evaluate: 
    if not os.path.exists(os.path.join(data_dir, session)):
        print(f"Session folder not found: {os.path.join(data_dir, session)}, skipping.")
        continue
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()


# put neural data through the pretrained model to get phoneme predictions (logits)
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []
        data['pred_seq'] = []
        data['losses'] = [] # calculate the loss per 
        
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)

            if eval_type == 'val' or 'test':
                logits_tensor = torch.tensor(logits, device=device)

                true_len = torch.tensor([data['seq_len'][trial]], device=device, dtype=torch.long)
                true_seq = torch.tensor(data['seq_class_ids'][trial][0:true_len], device=device, dtype=torch.long) # the integer phoneme sequence labels has a lot of trailing blanks which should be removed
                # length of each sequence of log_probs, for each sample in the batch
                adjusted_lens = torch.tensor([logits_tensor.shape[1]], device=device, dtype=torch.long)

                # Calculate the log probabilities of each phoneme label and reshapes (N,T,C) into (T,N,C)
                # C = number of characters in alphabet including blank, T = input length, and N = batch size
                log_probs = torch.permute(logits_tensor.log_softmax(2), [1, 0, 2])

                loss = ctc_loss(log_probs, true_seq, adjusted_lens, true_len)
                data['losses'].append(loss.item())
                
            pbar.update(1)
pbar.close()


# convert logits to phoneme sequences and print them out
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'phoneme_predictions_{time.strftime("%Y%m%d_%H%M%S")}.csv')

df = pd.DataFrame(columns=[
    "session", 
    "trial", 
    "true_phoneme", 
    "pred_phoneme", 
    "trial_ctc_loss",
    "trial_acc",
    "trial_ed", 
])

total_ed = 0
total_length_denom = 0

for session, data in test_data.items():
    data['pred_seq'] = []
    
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        # remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        # convert to phonemes
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        # add to data
        data['pred_seq'].append(pred_seq)

        # print out the predicted sequences
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val' or 'test':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

            print(f'Sentence label:      {sentence_label}')
            print(f'True sequence:       {"-".join(true_seq)}')

            # Calculate relevant metrics per trial
            pred_phonemes = '-'.join(pred_seq)
            true_phonemes = '-'.join(true_seq)

            ed = F.edit_distance(pred_seq, true_seq)
            total_ed += ed

            true_len = data['seq_len'][trial]
            total_length_denom += true_len
            assert true_len == len(true_phonemes.split("-")), f"{true_len} vs {len(true_phonemes.split('-'))}"
            
            new_row = pd.DataFrame([{
                "session": session,
                "trial": trial,
                "true_phoneme": true_phonemes,
                "pred_phoneme": pred_phonemes,
                "trial_ctc_loss": data['losses'][trial],
                "trial_acc": (1 - ed / true_len) if true_len > 0 else 0,
                "trial_ed": ed
            }])
            
            df = pd.concat([df, new_row], ignore_index=True)

            
        print(f'Predicted Sequence:  {" ".join(pred_seq)}')
        print()

print(f"Average PER = {total_ed/total_length_denom}")
df.to_csv(output_file, index=False)

print(f"Saved phoneme predictions, true sequences, and trial-level accuracy to {output_file}")