#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=1:0:0
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./bayesianV4Test/job_output.log
#SBATCH --error=./bayesianV4Test/job_error.log
#SBATCH --job-name=bayesianV4Test

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianV4Test25MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt --num_monte_carlo=25
srun python scitas.py 200 1 ./data ./bayesianV4Test50MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt --num_monte_carlo=50
srun python scitas.py 200 1 ./data ./bayesianV4Test75MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt --num_monte_carlo=75
srun python scitas.py 200 1 ./data ./bayesianV4Test100MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt --num_monte_carlo=100
srun python scitas.py 200 1 ./data ./bayesianV4Test200MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt --num_monte_carlo=200

deactivate