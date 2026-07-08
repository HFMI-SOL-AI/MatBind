#!/usr/bin/env bash
#SBATCH --account=solai
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --output=slurm/logs/infoNCE_%j.out
#SBATCH --error=slurm/logs/infoNCE_%j.err

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    MASTER_ADDR="$MASTER_ADDR"i
fi
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
# export MASTER_PORT=12345
export CUDA_VISIBLE_DEVICES=0,1,2,3

export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NCCL_IB_DISABLE=1

PROJECT_ROOT="/p/project1/solai/yang21/MatBind"
PYTHON_ENVIRONMENT="${PROJECT_ROOT}/matbind/bin/activate"
PYTHON_SCRIPT="${PROJECT_ROOT}/experiments/train.py"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)

cd "${PROJECT_ROOT}"
source "${PYTHON_ENVIRONMENT}"


srun --nodes=$SLURM_JOB_NUM_NODES \
  --gres=gpu:4 \
  --ntasks-per-node=4 \
  --cpus-per-task=4 \
  --cpu-bind=none \
  python ${PYTHON_SCRIPT}\
  "$@" \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  # encoders.use_gradient_checkpointing=true \
  # encoders.dos.encoder.use_gradient_checkpointing=false
