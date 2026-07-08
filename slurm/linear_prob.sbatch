#!/usr/bin/env bash
#SBATCH --account=solai
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition=develbooster
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err

export SRUN_CPUS_PER_TASK=4
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

export WANDB_PROJECT=linear_probe
export WANDB_ENTITY=leyang-forschungszentrum-j-lich

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}i"

# Load the necessary modules
module --force purge
module load Stages/2025  GCCcore/.13.3.0

srun --gres=gpu:4 --nodes=1 --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    export PYTHONPATH=''
    bash -c '
        cd /p/project1/solai/yang21/MatBind
        source matbind/bin/activate
        cd experiments/
        python linear_prob.py
    '
"