#!/bin/bash
#SBATCH --account=hai_solai
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=128
#SBATCH --job-name=preprocessing
#SBATCH --time=24:00:00
#SBATCH --partition=scraper
#SBATCH --nodes=1
source $HOME/.bashrc
echo "Running"

project1_ROOT=/p/project1/solai/oestreicher1/repos/MatBind

cd ${PROJECT_ROOT}

source ${PROJECT_ROOT}/.venv/bin/activate

srun python ${PROJECT_ROOT}/scripts/preprocess_crystal_structures.py

#salloc --partition=scraper --nodes=1 --account=hai_solai --time=04:00:00
# srun --cpu_bind=none --nodes=1 --pty /bin/bash -i
