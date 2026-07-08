#!/usr/bin/env bash
#SBATCH --account=solai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

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

PROJECT_ROOT="/p/project1/solai/oestreicher1/repos/MatBind"
PYTHON_ENVIRONMENT="${PROJECT_ROOT}/.venv/bin/activate"
PYTHON_SCRIPT="${PROJECT_ROOT}/experiments/retrieval_faiss.py"
cd "${PROJECT_ROOT}"
source "${PYTHON_ENVIRONMENT}"

srun --nodes=$SLURM_JOB_NUM_NODES \
  --gres=gpu:4 \
  --ntasks-per-node=4 \
  --cpus-per-task=4 \
  --cpu-bind=none \
  python ${PYTHON_SCRIPT} \
  # training_dir=/p/project1/solai/oestreicher1/repos/MatBind/outputs/train/2025-11-21/15-57-39
  # experiment=debug.yaml \
  # trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  # encoders.use_gradient_checkpointing=true \
  # encoders.dos.encoder.use_gradient_checkpointing=false
