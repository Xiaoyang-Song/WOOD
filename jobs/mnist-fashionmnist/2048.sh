#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=WDMFM2048
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDMFM2048.log

python3 main_OOD_binary.py 0.1 100 60 50 MNIST FashionMNIST 1 MNIST-FashionMNIST 2048 10