#!/bin/bash
#SBATCH --chdir=/scratch/izar/odefranc

#SBATCH --account=cs433
#SBATCH --qos=cs433

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1

#SBATCH --time=2:0:0  # Adjust based on expected test duration
#SBATCH --cpus-per-task=2

#SBATCH --mem-per-gpu=4G

#SBATCH --output=./output_quantile_v11/job_output_test.log
#SBATCH --error=./output_quantile_v11/job_error_test.log
#SBATCH --job-name=output_quantile_test_v11

module load gcc cuda python
source ./venvs/pytorch-env/bin/activate
srun python scitas.py 0 1 ./data ./output_quantile_v11 --normalized=True --skip_training=True --model_path=./output_quantile_v11/weights.pt
deactivate
