#!/bin/bash
#SBATCH --job-name=MYO_int32_all_index
#SBATCH --partition=mx
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --nodes=1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

module load crystfel/0.11.1

./indexamajig_python.py

