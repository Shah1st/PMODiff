#!/bin/bash -l

#SBATCH -o tune_c003
#SBATCH -J tune_c003
#SBATCH -p gpu-a-lowsmall,gpu-l40s,gpu-l40s-low,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=8000M
#STDIN=./infile

source ~/.bashrc
conda activate tagmol
#conda list
echo $CUDA_VISIBLE_DEVICES
cd "$SLURM_SUBMIT_DIR"
cd /mnt/scratch/users/andrij/PMODiff
python -m scripts.train_diffusion --config configs/tuning_c003.yml
#python -m scripts.train_diffusion --config configs/tuning_0_0.yml
#python -m scripts.train_diffusion --config configs/tuning_0_0.yml
#python -m scripts.train_diffusion --config configs/tuning_0_0.yml
