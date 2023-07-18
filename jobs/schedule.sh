#!/bin/bash

# Path Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Submit jobs

# FashionMNIST Within-Dataset Experiments
# sbatch jobs/fashionmnist/4.sh
# sbatch jobs/fashionmnist/8.sh
# sbatch jobs/fashionmnist/16.sh
# sbatch jobs/fashionmnist/32.sh
# sbatch jobs/fashionmnist/64.sh
# sbatch jobs/fashionmnist/128.sh
# sbatch jobs/fashionmnist/256.sh
# sbatch jobs/fashionmnist/512.sh
# sbatch jobs/fashionmnist/1024.sh
# sbatch jobs/fashionmnist/2048.sh
# sbatch jobs/fashionmnist/4096.sh

# CIFAR10-SVHN Between-Dataset Experiments
sbatch jobs/cifar10-svhn/4.sh
sbatch jobs/cifar10-svhn/8.sh
sbatch jobs/cifar10-svhn/16.sh
# sbatch jobs/cifar10-svhn/32.sh
# sbatch jobs/cifar10-svhn/64.sh
# sbatch jobs/cifar10-svhn/128.sh
# sbatch jobs/cifar10-svhn/256.sh
# sbatch jobs/cifar10-svhn/512.sh
# sbatch jobs/cifar10-svhn/1024.sh
# sbatch jobs/cifar10-svhn/2048.sh
sbatch jobs/cifar10-svhn/4096.sh