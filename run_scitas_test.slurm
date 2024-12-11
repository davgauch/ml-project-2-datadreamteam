#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=0:5:0
#SBATCH --cpus-per-task=2

#SBATCH --output=./bayesianVTempTest/job_output.log
#SBATCH --error=./bayesianVTempTest/job_error.log
#SBATCH --job-name=bayesianVTempTest

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianVTempTest --model_path=./bayesianV6Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=10 --skip_training=True

deactivate