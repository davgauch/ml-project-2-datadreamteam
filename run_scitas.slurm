#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=0:30:0
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./bayesianV3Test500MC/job_output.log
#SBATCH --error=./bayesianV3Test500MC/job_error.log
#SBATCH --job-name=bayesianV3Test500MC

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./bayesianV3Test500MC --bayesian=True --normalized=True --skip_training=True --model_path=./bayesianV3/weights.pt
deactivate