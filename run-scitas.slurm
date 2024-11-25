#!/bin/bash
#SBATCH --chdir /scratch/fcomte
#SBATCH --nodes 1
#SBATCH --time=2:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH -A cs433
#SBATCH -q cs433
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

module load gcc cuda python
source venvs/fcomte-pytorch/bin/activate
srun python scitas.py
deactivate