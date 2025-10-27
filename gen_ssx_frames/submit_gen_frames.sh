#!/bin/bash

# ===== Configuration =====
nframes=1000     # number of frames per h5, also determines the number of jobs
nthreads=20      # number of threads
chunksize=200    # number of frames stored in memory before writing
ntotal=40000     # total number of frames to process
sleep_time=0.2     # sleep time between submissions
python_script_path="../generate_h5.py"
# =========================

workdir=`pwd`
cp ${workdir}/${python_script_path} .
chmod +x ./*.py

# Compute number of iterations
niter=$((ntotal / nframes))

# Compute the ideal amount of memory
memory=$((nthreads * 4))

for ((i=0; i<niter; i++)); do
    # Compute N = nframes * i
    N=$((nframes * i))

    # Generate filename with zero-padded index (width 5)
    sbatch_file=$(printf "mb_sim_0.4p_0_%05d.sh" "$i")
    h5_filename=$(printf "mb_sim_0.4p_0_%05d.h5" "$i")
    jobname=$(printf "sim_0.4_%05d" "$i")
    logfile=$(printf "mb_sim_0.4p_0_%05d.log" "$i")

    cat >${sbatch_file} <<EOF
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --partition=nice,mx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${nthreads}
#SBATCH --time 3:00:00
#SBATCH --output=${jobname}.out
#SBATCH --error=${jobname}.err
#SBATCH --nodes=1
#SBATCH --mem=${memory}GB

# clean worker
conda deactivate >/dev/null 2>&1 || true
module purge

# load required modules
module load cuda
module load mamba
conda activate /gpfs/jazzy/data/scisoft/tfoldes/Python_venvs/nanoBragg_03

# run the script with the options
python3 generate_h5.py -t ${nthreads} \\
                       -f ${nframes} \\
                       -c ${chunksize} \\
                       -s ${N} \\
                       -o ${h5_filename} \\
                       -l ${logfile} \\
                       --force

EOF

    sbatch ${sbatch_file}

    # Sleep between iterations
    sleep "$sleep_time"
done


