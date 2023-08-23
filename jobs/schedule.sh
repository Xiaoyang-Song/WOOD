#!/bin/bash

# Path Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Submit jobs

# FashionMNIST Within-Dataset Experiments (DONE)
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

# CIFAR10-SVHN Between-Dataset Experiments (DONE)
# sbatch jobs/cifar10-svhn/4.sh
# sbatch jobs/cifar10-svhn/8.sh
# sbatch jobs/cifar10-svhn/16.sh
# sbatch jobs/cifar10-svhn/32.sh
# sbatch jobs/cifar10-svhn/64.sh
# sbatch jobs/cifar10-svhn/128.sh
# sbatch jobs/cifar10-svhn/256.sh
# sbatch jobs/cifar10-svhn/512.sh
# sbatch jobs/cifar10-svhn/1024.sh
# sbatch jobs/cifar10-svhn/2048.sh
# sbatch jobs/cifar10-svhn/4096.sh


# MNIST-FashionMNIST Between-Dataset Experiments
# sbatch jobs/mnist-fashionmnist/4.sh
# sbatch jobs/mnist-fashionmnist/8.sh
# sbatch jobs/mnist-fashionmnist/16.sh
# sbatch jobs/mnist-fashionmnist/32.sh
# sbatch jobs/mnist-fashionmnist/64.sh
# sbatch jobs/mnist-fashionmnist/128.sh
# sbatch jobs/mnist-fashionmnist/256.sh
# sbatch jobs/mnist-fashionmnist/512.sh
# sbatch jobs/mnist-fashionmnist/1024.sh
# sbatch jobs/mnist-fashionmnist/2048.sh
# sbatch jobs/mnist-fashionmnist/4096.sh

# MNIST Within-Dataset Experiments
# sbatch jobs/mnist/4.sh
# sbatch jobs/mnist/8.sh
# sbatch jobs/mnist/16.sh
# sbatch jobs/mnist/32.sh
# sbatch jobs/mnist/64.sh
# sbatch jobs/mnist/128.sh
# sbatch jobs/mnist/256.sh
# sbatch jobs/mnist/512.sh
# sbatch jobs/mnist/1024.sh
# sbatch jobs/mnist/2048.sh
# sbatch jobs/mnist/4096.sh


# SVHN Within-Dataset Experiments
# sbatch jobs/svhn/4.sh
# sbatch jobs/svhn/8.sh
# sbatch jobs/svhn/16.sh
# sbatch jobs/svhn/32.sh
# sbatch jobs/svhn/64.sh
# sbatch jobs/svhn/128.sh
# sbatch jobs/svhn/256.sh
# sbatch jobs/svhn/512.sh
# sbatch jobs/svhn/1024.sh
# sbatch jobs/svhn/2048.sh
# sbatch jobs/svhn/4096.sh

# FashionMNIST Within-Dataset Experiments (R2)
# sbatch jobs/fashionmnist-r2/4.sh
# sbatch jobs/fashionmnist-r2/8.sh
# sbatch jobs/fashionmnist-r2/16.sh
# sbatch jobs/fashionmnist-r2/32.sh
# sbatch jobs/fashionmnist-r2/64.sh
# sbatch jobs/fashionmnist-r2/128.sh
# sbatch jobs/fashionmnist-r2/256.sh
# sbatch jobs/fashionmnist-r2/512.sh
# sbatch jobs/fashionmnist-r2/1024.sh
# sbatch jobs/fashionmnist-r2/2048.sh
# sbatch jobs/fashionmnist-r2/4096.sh


# CIFAR10-SVHN (R2)
# sbatch jobs/cifar10-svhn-r2/4.sh
# sbatch jobs/cifar10-svhn-r2/8.sh
# sbatch jobs/cifar10-svhn-r2/16.sh
# sbatch jobs/cifar10-svhn-r2/32.sh
# sbatch jobs/cifar10-svhn-r2/64.sh
# sbatch jobs/cifar10-svhn-r2/128.sh
# sbatch jobs/cifar10-svhn-r2/256.sh
# sbatch jobs/cifar10-svhn-r2/512.sh
# sbatch jobs/cifar10-svhn-r2/1024.sh
# sbatch jobs/cifar10-svhn-r2/2048.sh
# sbatch jobs/cifar10-svhn-r2/4096.sh

# SVHN Within-Dataset Experiments (R2)
sbatch jobs/svhn-r2/4.sh
sbatch jobs/svhn-r2/8.sh
sbatch jobs/svhn-r2/16.sh
sbatch jobs/svhn-r2/32.sh
sbatch jobs/svhn-r2/64.sh
sbatch jobs/svhn-r2/128.sh
sbatch jobs/svhn-r2/256.sh
sbatch jobs/svhn-r2/512.sh
sbatch jobs/svhn-r2/1024.sh
sbatch jobs/svhn-r2/2048.sh
sbatch jobs/svhn-r2/4096.sh