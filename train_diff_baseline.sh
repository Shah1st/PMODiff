#!/bin/bash -l

#SBATCH -o slurm%j.out
#SBATCH -J baseline
#SBATCH -p gpu-h100 
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=8000M
#STDIN=./infile

source ~/.bashrc
conda activate tagmol
#conda list
echo $CUDA_VISIBLE_DEVICES
cd "$SLURM_SUBMIT_DIR"
python -m scripts.train_diffusion configs/training.yml
