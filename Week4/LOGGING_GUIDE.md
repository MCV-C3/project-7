# Week4 Directory Structure and Logging

## Complete Directory Structure

IMPORTANT: CHANGE mcvstudent29 by yours

All Week4 files are now organized under `/data/uabmcv2526/mcvstudent29/Week4/`:

```
/data/uabmcv2526/mcvstudent29/Week4/
│
├── logs/                           # SLURM stdout/stderr logs
│   └── run_train.sh_mcvstudent29_12345.out
│   └── run_train.sh_mcvstudent29_12345.err
│
├── output/                         # Training outputs
│   └── first_run/                  # experiment_name subfolder
│       ├── best_model.pt           # Best model checkpoint
│       ├── loss.png                # Loss curves plot
│       ├── accuracy.png            # Accuracy curves plot
│       └── training_summary.txt    # Training configuration & results
│
└── wandb/                          # Wandb offline logs
    └── offline-run-XXXXX/
        └── run-XXXXX.wandb
```

## What is `experiment_name`?

The `experiment_name` parameter (e.g., "first_run") is used to:

1. **Create a unique subfolder** for each experiment's outputs:
   - `/data/uabmcv2526/mcvstudent29/Week4/output/first_run/`
   
2. **Label the run in Wandb** so you can easily identify and compare experiments

3. **Organize multiple experiments** - when you change hyperparameters, use a different name:
   - `first_run` - baseline
   - `higher_lr` - experiment with higher learning rate
   - `more_dropout` - experiment with increased dropout
   - etc.

## Where Are Logs Saved?

### 1. SLURM Logs (stdout/stderr)
- **Location**: `/data/uabmcv2526/mcvstudent29/Week4/logs/`
- **What**: Console output, print statements, error messages
- **Filename format**: `{job_name}_{user}_{job_id}.out` and `.err`
- **Set by**: `#SBATCH -o` and `#SBATCH -e` in run_train.sh

### 2. Training Outputs (models, plots, summaries)
- **Location**: `/data/uabmcv2526/mcvstudent29/Week4/output/{experiment_name}/`
- **What**: 
  - `best_model.pt` - Model with best validation accuracy
  - `loss.png` - Training/test loss curves
  - `accuracy.png` - Training/test accuracy curves
  - `training_summary.txt` - Configuration and final results
- **Set by**: `--output_dir` argument in run_train.sh

### 3. Wandb Logs (experiment tracking)
- **Location**: `/data/uabmcv2526/mcvstudent29/Week4/wandb/`
- **What**: Detailed experiment logs (metrics per epoch, system info, etc.)
- **Set by**: `export WANDB_DIR=` in run_train.sh
- **Note**: In offline mode, sync later with `wandb sync`

## Example Usage

### Running Different Experiments

```bash
# Baseline experiment
python main.py \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name baseline \
    --learning_rate 1e-4 \
    --dropout 0.25

# Higher learning rate experiment
python main.py \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name high_lr_1e3 \
    --learning_rate 1e-3 \
    --dropout 0.25

# More regularization experiment
python main.py \
    --output_dir /data/uabmcv2526/mcvstudent29/Week4/output \
    --experiment_name more_dropout \
    --learning_rate 1e-4 \
    --dropout 0.5
```

Each experiment will create its own folder:
- `/data/.../Week4/output/baseline/`
- `/data/.../Week4/output/high_lr_1e3/`
- `/data/.../Week4/output/more_dropout/`

## Checking Your Results

After training:

1. **SLURM logs**: 
   ```bash
   cat /data/uabmcv2526/mcvstudent29/Week4/logs/*.out
   ```

2. **Model and plots**:
   ```bash
   ls /data/uabmcv2526/mcvstudent29/Week4/output/first_run/
   ```

3. **Wandb logs** (sync to cloud):
   ```bash
   wandb sync /data/uabmcv2526/mcvstudent29/Week4/wandb/offline-run-*
   ```

## Summary

✅ All Week4 files are now in one place: `/data/uabmcv2526/mcvstudent29/Week4/`

✅ Each experiment gets its own subfolder based on `experiment_name`

✅ Easy to compare different experiments by checking different experiment folders

✅ Clean organization matching Week3 structure
