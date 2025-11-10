# Training and Evaluating Minimal Brain-To-Text

Basic pipeline which trains and evaluates a model with 2 python commands.

When evaluating, you can use `--phoneme_predictions_csv` to directly analyze existing predictions.

```
import os

# Train a model
! python /kaggle/working/nejm-brain-to-text-main-syde577/scripts/minimal_train.py --config /kaggle/working/nejm-brain-to-text-main-syde577/scripts/minimal_baseline_args.yaml

# Evaluate a model
repo_dir = "/kaggle/working/nejm-brain-to-text-main"
eval_script = "scripts/minimal_evaluate.py"
model_path = "/kaggle/working/output/checkpoints"
data_dir = "/kaggle/input/brain-to-text-25-minimal/t15_copyTask_neuralData/hdf5_data_final"
csv_path = "/kaggle/working/nejm-brain-to-text-main-syde577/data/t15_copyTaskData_description.csv"
eval_type = "test"  # "val" or "test"
gpu_number = 0     # GPU 0, -1 for CPU

cmd = f"""
cd /kaggle/working/nejm-brain-to-text-main-syde577 && \

python {eval_script} \
    --model_path {model_path} \
    --data_dir {data_dir} \
    --csv_path {csv_path} \
    --eval_type {eval_type} \
    --gpu_number {gpu_number}
"""
print("Running evaluation script...")
os.system(cmd)
```