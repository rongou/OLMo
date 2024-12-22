#!/bin/bash

#SBATCH --account=sw_aidot
#SBATCH --chdir=/lustre/fsw/portfolios/sw/users/rou/src/OLMo
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --job-name=olmo-1b
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --output=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.out
#SBATCH --partition=batch_short
#SBATCH --time=0-2

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_NET_CHUNKSIZE=2097152

# Initialize micromamba
source /home/rou/.bashrc
micromamba activate olmo

# Set environment variables for CUDA and OpenMP
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/olmo
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=16

# Set environment variables for distributed training
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + ${SLURM_JOBID: -4}))
export MASTER_ADDR
export MASTER_PORT
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"

# Launch OLMo training
srun python \
  scripts/train.py \
  configs/official-0724/OLMo-1B.yaml \
  --run_name="$SLURM_JOB_NAME"
