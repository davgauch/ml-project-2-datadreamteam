#!/bin/bash
#SBATCH --chdir=/scratch/izar/fcomte

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=6:0:0
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./output_bayesian_with_earlystopperv2/job_output.log
#SBATCH --error=./output_bayesian_with_earlystopperv2/job_error.log
#SBATCH --job-name=bayesian_with_earlystopperv2

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./output_bayesian_with_earlystopperv2 --bayesian=True
deactivate