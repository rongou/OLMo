#!/bin/bash

#SBATCH --job-name=olmo-1b-baseline
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --network=ib
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.err
#SBATCH --account=sw_aidot
#SBATCH --partition=batch_short
#SBATCH --chdir=/lustre/fsw/portfolios/sw/users/rou/src/OLMo

# Initialize micromamba
source /home/rou/.bashrc
micromamba activate olmo

# Create a unique directory for this job in /raid/scratch
export JOB_SCRATCH_DIR="/raid/scratch/$USER/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH_DIR"

# Set cache directories to local raid storage
export HF_HOME="$JOB_SCRATCH_DIR/hf_cache"
export WANDB_DIR="$JOB_SCRATCH_DIR/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"

# Set environment variables for CUDA and OpenMP
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/olmo
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=16

# Set environment variables for distributed training
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

# Set environment variables for NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5

# Launch OLMo training
srun --cpu-bind=cores --gpu-bind=closest \
    python \
    scripts/train.py \
    configs/official-0724/OLMo-1B.yaml \
    --run_name="$SLURM_JOB_NAME"

# Cleanup scratch directory after job completes
rm -rf "$JOB_SCRATCH_DIR"
