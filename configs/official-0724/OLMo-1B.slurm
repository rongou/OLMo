#!/bin/bash
#SBATCH --job-name=olmo-1b-baseline
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --network=infiniband
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/sw/users/rou/logs/%x_%j.err
#SBATCH --account=sw_aidot
#SBATCH --partition=batch_short
#SBATCH --signal=SIGTERM@90 # Get notified 90 seconds before preemption
#SBATCH --chdir=/lustre/fsw/portfolios/sw/users/rou/src/OLMo

# Initialize micromamba
source /home/rou/.bashrc

# Activate olmo environment
micromamba activate olmo

# Create a unique directory for this job in /raid
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
export TORCH_CUDA_ARCH_LIST="hopper"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Set the master address and port for torch.distributed
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Enable Infiniband for NCCL
export NCCL_IB_DISABLE=0

# Launch OLMo training using torchrun
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    scripts/train.py \
    configs/official-0724/OLMo-1B-Slurm.yaml \
    --run_name="$SLURM_JOB_NAME"

# Cleanup scratch directory after job completes
rm -rf "$JOB_SCRATCH_DIR"
