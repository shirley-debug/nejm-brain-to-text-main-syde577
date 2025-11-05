import os
import shutil
import glob
import pandas as pd
import h5py
import json 
import numpy as np

OUTPUT_DIR = "/kaggle/working/brain-to-text-25-minimal-shirley/t15_copyTask_neuralData/hdf5_data_final"
SRC_DIR = "/kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SESSIONS_INCLUDED = {'t15.2023.08.13', 't15.2023.08.18', 't15.2023.08.20', 't15.2023.08.11'}
BLOCK_VAL_TEST_SPLIT = {
    't15.2023.08.13': [8, 9],
    't15.2023.08.18': [6, 7],
    't15.2023.08.20': [7, 8],
}

### HELPER FUNCTIONS ###

def save_h5py_file(file_path, data):
    """Save data to HDF5 file matching original format exactly."""
    with h5py.File(file_path, 'w') as f:
        num_trials = len(data['neural_features'])
        
        for i in range(num_trials):
            trial_name = f"trial_{i:04d}"
            g = f.create_group(trial_name)
            
            # Save datasets (keep exact same format as original)
            g.create_dataset('input_features', data=data['neural_features'][i])
            
            if data['seq_class_ids'][i] is not None:
                g.create_dataset('seq_class_ids', data=data['seq_class_ids'][i])
            
            # Transcription is int32, same as seq_class_ids
            if data['transcriptions'][i] is not None:
                g.create_dataset('transcription', data=data['transcriptions'][i])
            
            # Save scalar attributes
            g.attrs['n_time_steps'] = data['n_time_steps'][i]
            if data['seq_len'][i] is not None:
                g.attrs['seq_len'] = data['seq_len'][i]
            if data['sentence_label'][i] is not None:
                g.attrs['sentence_label'] = data['sentence_label'][i]
            
            g.attrs['session'] = data['session'][i]
            g.attrs['block_num'] = data['block_num'][i]
            g.attrs['trial_num'] = data['trial_num'][i]
            

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data


### END HELPER FUNCTIONS ###

print("Copying just the first 4 sessions...")
all_dirs = sorted(
    [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
)
dirs_to_copy = all_dirs[:4]

for d in dirs_to_copy:
    src_path = os.path.join(SRC_DIR, d)
    dst_path = os.path.join(OUTPUT_DIR, d)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    print(f"Copied {src_path} → {dst_path}")

print(f"\nCopied {len(dirs_to_copy)} directories.")


# Leave train as is, delete test data
print("\nDelete all the test data which doesn't have sentence labels...")
pattern = os.path.join(OUTPUT_DIR, "**", "*test*")

for file_path in glob.glob(pattern, recursive=True):
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Get all files to work with
files = glob.glob(os.path.join(OUTPUT_DIR, "**", "*"), recursive=True)
files = [f for f in files if os.path.isfile(f)]

sessions_included = set([os.path.basename(os.path.dirname(f)) for f in files])
print(f"Sessions Included = {sessions_included}")
assert SESSIONS_INCLUDED == sessions_included

# Split validation data into val and test
for session, val_test_split in BLOCK_VAL_TEST_SPLIT.items():
    for file in files:
        # This is the ORIGINAL val data, split into val and test hdf5 files
        if f"{session}/data_val.hdf5" in file:
            val_data = {
                'neural_features': [],
                'n_time_steps': [],
                'seq_class_ids': [],
                'seq_len': [],
                'transcriptions': [],
                'sentence_label': [],
                'session': [],
                'block_num': [],
                'trial_num': [],
            }
            test_data = {
                'neural_features': [],
                'n_time_steps': [],
                'seq_class_ids': [],
                'seq_len': [],
                'transcriptions': [],
                'sentence_label': [],
                'session': [],
                'block_num': [],
                'trial_num': [],
            }
            
            data = load_h5py_file(file)
            num_blocks = len(data['block_num'])
            print(f"Number of blocks in {session} is {num_blocks}.")
            
            for key, val in data.items():
                assert len(val) == num_blocks # ensure there's an element for every block

            for idx, block in enumerate(data['block_num']):
                if block == val_test_split[0]: # allocate to validation
                    val_data['neural_features'].append(data['neural_features'][idx]) 
                    val_data['n_time_steps'].append(data['n_time_steps'][idx])
                    val_data['seq_class_ids'].append(data['seq_class_ids'][idx])
                    val_data['seq_len'].append(data['seq_len'][idx])
                    val_data['transcriptions'].append(data['transcriptions'][idx])
                    val_data['sentence_label'].append(data['sentence_label'][idx])
                    val_data['session'].append(data['session'][idx])
                    val_data['block_num'].append(data['block_num'][idx])
                    val_data['trial_num'].append(data['trial_num'][idx])
                elif block == val_test_split[1]: # allocate to test
                    test_data['neural_features'].append(data['neural_features'][idx]) 
                    test_data['n_time_steps'].append(data['n_time_steps'][idx])
                    test_data['seq_class_ids'].append(data['seq_class_ids'][idx])
                    test_data['seq_len'].append(data['seq_len'][idx])
                    test_data['transcriptions'].append(data['transcriptions'][idx])
                    test_data['sentence_label'].append(data['sentence_label'][idx])
                    test_data['session'].append(data['session'][idx])
                    test_data['block_num'].append(data['block_num'][idx])
                    test_data['trial_num'].append(data['trial_num'][idx])
                else:
                    raise Exception("Invalid block number.")

            print(f"Saving hdf5 file to {os.path.join(os.path.dirname(file), 'data_val.hdf5')}")
            save_h5py_file(os.path.join(os.path.dirname(file), "data_val.hdf5"), val_data)
            
            print(f"Saving hdf5 file to {os.path.join(os.path.dirname(file), 'data_test.hdf5')}")
            save_h5py_file(os.path.join(os.path.dirname(file), "data_test.hdf5"), test_data)