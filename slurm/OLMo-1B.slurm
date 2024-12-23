#!/bin/bash

#SBATCH --job-name=olmo-1b
#SBATCH --dependency=singleton
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=sw_aidot
#SBATCH --partition=batch_short
#SBATCH --time=2
#SBATCH --output=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.out
#SBATCH --chdir=/lustre/fsw/portfolios/sw/users/rou/src/OLMo

srun -l direct.sh
