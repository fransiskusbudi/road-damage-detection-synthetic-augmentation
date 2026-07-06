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

SCRATCH_DISK=/disk/scratch_big
dest_path=${SCRATCH_DISK}/${USER}/rdd-YOLO
src_path=/home/${USER}/rdd-YOLO/
results_path=/home/${USER}/rdd-YOLO/results

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

# Copy project files excluding large directories
rsync -azvP --exclude 'runs/' ${src_path} ${dest_path}

# Copy only the relevant dataset subfolder
DATASET_NAME=${1:-"baseline"}  # Default to "baseline" if not provided as an argument
dataset_src="${src_path}datasets/${DATASET_NAME}"
dataset_dest="${dest_path}/datasets/${DATASET_NAME}"

if [ -d "${dataset_src}" ]; then
    echo "Copying dataset: ${DATASET_NAME}"
    mkdir -p "${dataset_dest}"
    rsync -azvP "${dataset_src}/" "${dataset_dest}/"
else
    echo "❌ ERROR: Dataset folder '${dataset_src}' does not exist."
    echo "Available datasets:"
    ls "${src_path}datasets/"
    exit 1
fi

# Change directory
cd ${dest_path}
echo "Changed directory to ${dest_path}"

# Run the script
python trial_baseline_new.py

# ====================
# Move output data back to Home Directory
# ====================
echo "Moving output data back to Home Directory"
mkdir -p ${results_path}
rsync -azvP ${dest_path}/outputs/ ${results_path}/  # ✅ Fixed path

# ====================
# Clean up scratch space
# ====================
echo "Deleting scratch disk path: ${dest_path}"
rm -rf ${dest_path}

echo ""
echo "============"
echo "Job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"


