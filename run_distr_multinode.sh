#!/bin/bash
#SBATCH --job-name=distr_multi
#SBATCH --partition=a2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=distr_single_%j.out
#SBATCH --error=distr_single_%j.err

eval "$(conda shell.bash hook)"
conda activate cs336_systems

export MASTER_PORT=$(expr 10000 + $(echo -n $Slurm_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$Slurm_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

srun python3 cs336-systems/cs336_systems/distributed_multinode.py --num_warmup_steps 3 --num_trial_steps 1
mkdir -p logs/output
mkdir -p logs/error
#mv profiling*.out logs/output/.
#mv profiling*err logs/error/.
#mv lm_profiler_stacks.txt logs/output/.
mv distr*out logs/output/.
mv distr*err logs/error/.
