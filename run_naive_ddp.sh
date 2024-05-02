#!/bin/bash
#SBATCH --job-name=ddp
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=a2
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err

eval "$(conda shell.bash hook)"
conda activate cs336_systems

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

srun python3 cs336-systems/cs336_systems/naive_ddp.py --backend 'nccl' --device 'gpu'

mkdir -p logs/output
mkdir -p logs/error

mv distr*out logs/output/.
mv distr*err logs/error/.
