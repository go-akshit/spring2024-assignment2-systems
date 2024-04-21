#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --partition=a2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=profiling_%j.out
#SBATCH --error=profiling_%j.err
#SBATCH --gpus=1
python3 cs336-systems/cs336_systems/benchmarking.py --context_length=10 --vocab_size=10 --batch_size=2 --num_heads=2 --num_layers=2 --warmup_steps=0 --measurement_steps=3 --pass_type='both' --profiler=True
mkdir -p logs/output
mkdir -p logs/error
mv profiling*.out logs/output/.
mv profiling*err logs/error/.
mv lm_profiler_stacks.txt logs/output/.
