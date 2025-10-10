#!/bin/bash

#SBATCH --job-name=AcNiR_2.0_ambi
#SBATCH --partition=mx-low
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time 72:00:00
#SBATCH --nodes=1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

hostname

module purge
module load crystfel/0.11.1 >/dev/null
module load ccp4 >/dev/null
module load xds >/dev/null

# run the ambigator script
./ambigator_master.py -i ../run_dense_first.stream \
                      -p P213 \
                      -O k,h,-l \
                      -j 20



