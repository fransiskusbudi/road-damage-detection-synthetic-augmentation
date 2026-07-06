#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast
# Activate conda environment
echo "Activating Conda environment..."
source /home/${USER}/miniconda3/bin/activate wgan

# Set CUDA devices for multi-GPU training

# Print GPU info
echo "Using GPUs:"
nvidia-smi

# Run the training script
echo "Starting WGAN-GP Training..."
python main.py 

# Print completion message
echo "Training complete!"
