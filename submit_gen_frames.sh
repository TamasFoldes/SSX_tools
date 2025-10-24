#!/bin/bash
#SBATCH --job-name=gen_0_4_p
#SBATCH --partition=mx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time 72:00:00
#SBATCH --nodes=1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

conda deactivate
module purge
module load cuda
module load mamba
conda activate /gpfs/jazzy/data/scisoft/tfoldes/Python_venvs/nanoBragg_03

printf "" > h5_generation.log

# ===== Configuration =====
nframes=1000     # number of frames; also the scaling factor for N
nthreads=40      # number of computational threads
chunksize=200    # number of frames generated at once before writing
ntotal=40000     # total number of frames to process
sleep_time=1     # seconds to sleep between iterations
# =========================

# Compute number of iterations
niter=$((ntotal / nframes))

echo "Running $niter iterations (ntotal=$ntotal, nframes=$nframes)..."

for ((i=0; i<niter; i++)); do
    # Compute N = nframes * i
    N=$((nframes * i))

    # Generate filename with zero-padded index (width 5)
    filename=$(printf "mb_sim_0.4p_0_%05d.h5" "$i")

    # Run the Python command
    python3 generate_h5.py -t "$nthreads" -f "$nframes" -c "$chunksize" -s "$N" -o "$filename" --force

    # Sleep between iterations
    sleep "$sleep_time"
done
