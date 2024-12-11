#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=10:0:0
#SBATCH --cpus-per-task=2

#SBATCH --output=./bayesianV7Train/job_output.log
#SBATCH --error=./bayesianV7Train/job_error.log
#SBATCH --job-name=bayesianV7Train

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianV7Train --bayesian=True --normalized=True --num_monte_carlo=50

deactivate