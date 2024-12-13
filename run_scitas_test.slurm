#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=0:30:0
#SBATCH --cpus-per-task=2

#SBATCH --output=./bayesianV8Test/job_output.log
#SBATCH --error=./bayesianV8Test/job_error.log
#SBATCH --job-name=bayesianV8Test

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianV8Test5 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=5 --skip_training=True
srun python scitas.py 200 1 ./data ./bayesianV8Test10 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=10 --skip_training=True
srun python scitas.py 200 1 ./data ./bayesianV8Test20 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=20 --skip_training=True
srun python scitas.py 200 1 ./data ./bayesianV8Test30 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=30 --skip_training=True
srun python scitas.py 200 1 ./data ./bayesianV8Test40 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=40 --skip_training=True
srun python scitas.py 200 1 ./data ./bayesianV8Test50 --model_path=./bayesianV8Train/weights.pt --bayesian=True --normalized=True --num_monte_carlo=50 --skip_training=True

deactivate