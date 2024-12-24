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
#SBATCH --time=0-2
#SBATCH --output=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.out
#SBATCH --chdir=/lustre/fsw/portfolios/sw/users/rou/src/OLMo

function run_training() {
  source /home/rou/.bashrc
  micromamba activate olmo

  set -euxo pipefail

  # Set environment variables for CUDA and OpenMP
  export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/olmo
  export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

  # setup pytorch distributed
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT=6000

  # add some magic incantations
  export NCCL_IB_SL=1
  export NCCL_IB_TIMEOUT=19
  export UB_TIMEOUT=720
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NVTE_FWD_LAYERNORM_SM_MARGIN=16
  export NVTE_BWD_LAYERNORM_SM_MARGIN=16
  export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
  export NCCL_P2P_NET_CHUNKSIZE=2097152
  export NCCL_DEBUG=WARN

  export NCCL_SHM_DISABLE=1
  export NCCL_PROTO=simple
  export NCCL_NVLS_ENABLE=0

  export NCCL_IB_DISABLE=0
  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8
  export NCCL_IB_GID_INDEX=3
  export NCCL_IB_TC=96
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export MELLANOX_VISIBLE_DEVICES=0,1,2,3,5,6,7,8
  export NCCL_SOCKET_IFNAME=enp90s0np0

  # ignore some warnings
  export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

  # launch trainer directly
  cd /lustre/fsw/portfolios/sw/users/rou/src/OLMo
  python scripts/train.py \
      configs/official-0724/OLMo-1B.yaml \
      --run_name="$SLURM_JOB_NAME"
}

export -f run_training
srun -l bash -c 'run_training'
