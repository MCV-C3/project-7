#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p mhigh
#SBATCH -q masterlow
#SBATCH --mem=4096
#SBATCH --gres=gpu:1
#SBATCH --job-name=week2
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err

RUN_DIR=/export/home/group07/week2/runs/${SLURM_JOB_NAME}_${SLURM_JOB_USER}_${SLURM_JOB_ID}/

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

sleep 5

 # Load conda environment and run the script
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c3
python /export/home/group07/week2/main.py "$RUN_DIR"
