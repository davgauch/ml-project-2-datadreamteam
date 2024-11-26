#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

#SBATCH --ntasks=1

#SBATCH --time=4:0:0
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./output_bayesianV1/job_output.log
#SBATCH --error=./output_bayesianV1/job_error.log
#SBATCH --job-name=bayesianV1

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./output_bayesianV1 --bayesian=True
deactivate