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

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate pt

SCRATCH_DISK=/disk/scratch
dest_path=${SCRATCH_DISK}/${USER}/mlpractical_rdd/

# ====================
# Clean up scratch space
# ====================
if [ -d "${dest_path}" ]; then
    echo "Deleting scratch disk path: ${dest_path}"
    rm -rf "${dest_path}"
else
    echo "Scratch disk path does not exist: ${dest_path}"
fi

mkdir -p ${dest_path}
src_path=/home/${USER}/mlpractical_rdd/diffusion_gan
rsync -azvP --exclude='outputs/' \
            --exclude='dataset/' \
            "${src_path}" "${dest_path}"

# --exclude='bdd_diffusion_100' \
python trial_baseline_new.py

# echo "Moving output data back to Home"

# rsync -azvP ${dest_path}runs/ ${src_path}/results/

# ====================
# Clean up scratch space
# ====================
echo "Deleting scratch disk path: ${dest_path}"
rm -rf ${dest_path}


echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
