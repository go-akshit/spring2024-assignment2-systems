#!/bin/bash
#SBATCH --job-name=distr_multi
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00

eval "$(conda shell.bash hook)"
conda activate cs336_systems

export MASTER_PORT=$(expr 10000 + $(echo -n $Slurm_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$Slurm_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

srun python3 cs336-systems/cs336_systems/distributed_multinode.py --num_warmup_steps 3 --num_trial_steps 1
