#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=WDMFM64
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDMFM64.log

python3 main_OOD_binary.py 0.1 100 60 50 MNIST FashionMNIST 1 MNIST-FashionMNIST 64 10