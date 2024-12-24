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
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT=6000

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

  # launch trainer directly
  cd /lustre/fsw/portfolios/sw/users/rou/src/OLMo
  python -W ignore::FutureWarning scripts/train.py \
      configs/official-0724/OLMo-1B.yaml \
      --run_name="$SLURM_JOB_NAME"
}

export -f run_training
srun -l bash -c 'run_training'
