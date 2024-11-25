#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --time=2:0:0

#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py
deactivate