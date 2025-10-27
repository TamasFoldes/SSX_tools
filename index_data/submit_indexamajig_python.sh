#!/bin/bash
#SBATCH --job-name=MYO_int32_all_index
#SBATCH --partition=mx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time 72:00:00
#SBATCH --nodes=1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

module purge
module load crystfel

./indexamajig_python.py

