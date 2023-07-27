#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=WDFM4
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDFM4.log

python3 main_OOD_binary.py 0.1 100 60 50 FashionMNIST-17 FashionMNIST-89 1 FashionMNIST 4 8