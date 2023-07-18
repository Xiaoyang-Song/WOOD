#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=WDCS32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=32
#SBATCH --time=144:00:00
#SBATCH --output=/home/xysong/WOOD/slurm-jobs/WDCS32.log

python3 main_OOD_binary.py 0.1 100 60 50 Cifar10 SVHN 3 CIFAR10-SVHN 32 10
