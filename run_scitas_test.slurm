#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=0:30:0
#SBATCH --cpus-per-task=2

#SBATCH --output=./bayesianV5Test/job_output.log
#SBATCH --error=./bayesianV5Test/job_error.log
#SBATCH --job-name=bayesianV5Test

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianV5Test --model_path=./bayesianV5Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=50 --skip_training=True

deactivate