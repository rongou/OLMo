#!/bin/bash

source /home/rou/.bashrc
micromamba activate olmo

set -euxo pipefail

# Set environment variables for CUDA and OpenMP
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/olmo
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# setup pytorch distributed
WORLD_SIZE=$SLURM_NTASKS
RANK=$SLURM_PROCID
LOCAL_RANK=$SLURM_LOCALID
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000
export WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

# add some magic incantations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_PROTO=simple
export NCCL_SHM_DISABLE=1
export UB_TIMEOUT=720

python scripts/train.py --run_name="$SLURM_JOB_NAME"
