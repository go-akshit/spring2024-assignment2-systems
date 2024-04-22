#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --partition=a2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=profiling_%j.out
#SBATCH --error=profiling_%j.err
#SBATCH --gpus=1
#SBATCH --mem=200G
#python3 cs336-systems/cs336_systems/benchmarking.py --context_length=128 --vocab_size=10000 --batch_size=16 --num_heads=12 --num_layers=12 --warmup_steps=1 --measurement_steps=5 --pass_type='both' --norm_layer='ln'
python3 cs336-systems/cs336_systems/rms_layernorm.py
mkdir -p logs/output
mkdir -p logs/error
mv profiling*.out logs/output/.
mv profiling*err logs/error/.
mv lm_profiler_stacks.txt logs/output/.
