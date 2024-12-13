#!/bin/bash
#SBATCH --chdir=/scratch/izar/odefranc

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=10:0:0
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./output_quantile_v10/job_output.log
#SBATCH --error=./output_quantile_v10/job_error.log
#SBATCH --job-name=output_quantile_v10

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 200 1 ./data ./output_quantile_v10 --normalized=True
deactivate