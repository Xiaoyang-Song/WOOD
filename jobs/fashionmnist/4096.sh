#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=WDFM4096
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDFM4096.log

python3 main_OOD_binary.py 0.1 100 60 50 Cifar10 SVHN 3 CIFAR10-SVHN 4096 10