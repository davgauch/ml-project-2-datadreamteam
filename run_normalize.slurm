#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --ntasks 1

#SBATCH --time 2:0:0
#SBATCH --cpus-per-task 2

#SBATCH --mem-per-gpu 64G

#SBATCH --output=./normalizationM/job_output.log
#SBATCH --error=./normalizationM/job_error.log
#SBATCH --job-name=normalizatioM

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python normalize_datasets.py
deactivate