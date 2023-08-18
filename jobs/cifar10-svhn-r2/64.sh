#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=WDCS64
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=36:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDCS64-R2.log

python3 main_OOD_binary.py 0.1 100 60 50 Cifar10 SVHN 3 CIFAR10-SVHN 64 10
