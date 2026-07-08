#!/usr/bin/env bash

# Check if experiment name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

# List of L1 norm weight steps
STEP_LIST=(0.001 0.01 0.1 1)

# Loop through steps
for l1_weight in "${STEP_LIST[@]}"
do
    cat << EOF > submit_job.sbatch
#!/usr/bin/env bash
#SBATCH --account=westai0036
#SBATCH --job-name=${1}_l1_${l1_weight}
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=dc-hwai
#SBATCH --mem-per-cpu=32G
#SBATCH --output=run_out-distrib_l1_${l1_weight}.%j
#SBATCH --error=run_err-distrib_l1_${l1_weight}.%j

export SRUN_CPUS_PER_TASK=4
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

MASTER_ADDR="\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="\${MASTER_ADDR}i"

module --force purge
echo ${1}
srun --nodes=1 --gres=gpu:4 --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    export PYTHONPATH=''
    bash -c '
        source /p/project1/hai_molbind/miniconda3/bin/activate
        conda activate matbind_test
        python train.py 'experiment=${1}' 'trainer.num_nodes=\$SLURM_JOB_NUM_NODES' 'data.num_workers=1' 'model.loss.sparse=True' 'model.loss.l1_norm_weight=${l1_weight}' 'run_id=testing_l1_${l1_weight}'
    '
"
EOF

    sbatch submit_job.sbatch
done
