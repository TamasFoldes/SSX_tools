#!/usr/bin/env python3

import sys
import logging
import functools
import numpy as np
import subprocess
import glob
import os
import re
import h5py as h5

log_filename = "indexamajig.log"
log_level = logging.INFO


def main():
    logging.getLogger(__name__).setLevel(logging.INFO)

    h5_location = "/gpfs/ga/data/visitor/mx2633/id29/20250328/RAW_DATA/Mb_data_types/int32_925Hz/run_01_ssx_foil_collection/Mb-Mb*.h5"
    hitfile = "hits.lst"
    indexamajig_out_filename = "indexamajig_output.log"
    indexamajig_err_filename = "indexamajig_error.log"
    geom_file = "../jungfrau4m-optimized.geom"
    cell_file = "../myoglobin_full.cell"

    get_selected_hits(pattern=h5_location,
                      hitfile=hitfile,
                      onlyhits=False)

    indexamajig_command = f"""indexamajig --geometry {geom_file} \
                                          --input {hitfile} \
                                          --pdb {cell_file} \
                                          --output run_dense_first.stream \
                                          --peaks peakfinder8 \
                                          --min-peaks 10 \
                                          --multi \
                                          --peak-radius=4.0,6.0,10.0 \
                                          --min-pix-count=3 \
                                          --min-snr=4 \
                                          --threshold=800 \
                                          --local-bg-radius=10 \
                                          --indexing xgandalf,mosflm \
                                          --int-radius=4.0,6.0,10.0 \
                                          --no-non-hits-in-stream \
                                          --no-retry \
                                          --xgandalf-fast-execution \
                                          -j 20"""

    run_indexamajig(indexamajig_command,
                    out_filename=indexamajig_out_filename,
                    err_filename=indexamajig_err_filename)
    get_indexing_statistics(filename=indexamajig_err_filename)


class StoreOutput:
    """Custom stream class to write to both stdout and a log file."""

    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream  # Original sys.stdout or sys.stderr

    def write(self, message):
        self.stream.write(message)  # Write to standard output
        self.log_file.write(message)  # Write to the log file

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def log_to_file(log_filename, log_level=log_level):
    """Decorator to log function output (stdout & stderr) to a file and also keep printing to console."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Configure logging
            logging.basicConfig(
                filename=log_filename,
                level=log_level,
                format="[%(asctime)s.%(msecs)03d] - %(levelname)-8s - %(message)s",
                datefmt="%Y-%b-%d %H:%M:%S",
                filemode="w"
            )

            with open(log_filename, "a") as log_file:
                # Save original stdout and stderr
                original_stdout = sys.stdout
                original_stderr = sys.stderr

                try:
                    # Create a StoreOutput stream to duplicate stdout and stderr
                    sys.stdout = StoreOutput(log_file, original_stdout)
                    sys.stderr = StoreOutput(log_file, original_stderr)

                    logging.debug(
                        f"Calling function: {func.__name__} with args: {args}, kwargs: {kwargs}")
                    result = func(*args, **kwargs)
                    logging.debug(
                        f"Function {func.__name__} returned: {result}")

                    return result

                except Exception as e:
                    logging.error(
                        f"Error in function {func.__name__}: {e}", exc_info=True)
                    raise e  # Re-raise the exception

                finally:
                    # Restore original stdout and stderr
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

        return wrapper

    return decorator


@log_to_file(log_filename, log_level=log_level)
def get_selected_hits(pattern, Nsamples=None, rseed=42, Nmax=None, hitfile=None, onlyhits=True):

    @log_to_file(log_filename, log_level=logging.INFO)
    def find_files_by_pattern(pattern):
        logging.info(f"Searching for files with pattern:\n {pattern}")
        filenames = glob.glob(pattern)
        filenames.sort()
        return filenames

    @log_to_file(log_filename, log_level=logging.INFO)
    def collect_allhits(filenames, onlyhits=True):
        logging.info("Collecting hits")
        all_hits = []
        for filename in filenames:
            with h5.File(filename, "r") as h5file:
                dset_path = "/entry_0000/measurement/data"
                ishit_path = "/entry_0000/processing/peakfinder/isHit"
                for i in range(len(h5file[dset_path])):
                    if onlyhits == False:
                        all_hits.append("%s //%d\n" % (filename, i))
                    if onlyhits == True:
                        if h5file[ishit_path][i] == 1:
                            all_hits.append("%s //%d\n" % (filename, i))
        return all_hits

    np.random.seed(rseed)
    logging.debug(f"Random seed: {rseed}")
    filenames = find_files_by_pattern(pattern)
    if Nmax is not None:
        logging.info(f"Limiting number of files to {Nmax}")
        filenames = filenames[:Nmax]
    all_hits = collect_allhits(filenames, onlyhits=onlyhits)
    if Nsamples is None:
        selected_hits = all_hits
        logging.info("Number of hits collected: %d" % len(selected_hits))
    else:
        index_list = np.sort(np.random.choice(
            a=len(all_hits), size=Nsamples, replace=False))
        selected_hits = [all_hits[i] for i in index_list]
        logging.info("Number of hits collected: %d" % len(selected_hits))
    if hitfile is not None:
        with open(hitfile, "w") as ofile:
            logging.info(f"Writing selected hits to file {hitfile}")
            ofile.writelines(selected_hits)
    else:
        return selected_hits


@log_to_file(log_filename, log_level=log_level)
def run_indexamajig(indexamajig_command,
                    out_filename="indexamajig_output.log",
                    err_filename="indexamajig_error.log"):

    def get_formatted_command(text, num_spaces=4):
        formatted_text = re.sub(r'\s{2,}', '\n', text)
        lines = formatted_text.splitlines()
        indentation = " " * num_spaces
        return "\n".join([lines[0]] + [indentation + line for line in lines[1:]])

    formatted_command = get_formatted_command(
        indexamajig_command, num_spaces=12)
    logging.info(f"Running indexamajig with command:\n{formatted_command}")
    with open(out_filename, "w") as out, open(err_filename, "w") as err:
        with subprocess.Popen(['/bin/bash', '-c', indexamajig_command],
                              stdout=out,
                              stderr=err,
                              cwd=os.getcwd(),
                              text=True
                              ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logging.error("indexamajig failed")
                quit()
            else:
                logging.info("Indexamajig finished successfully")


@log_to_file(log_filename, log_level=log_level)
def get_indexing_statistics(filename):
    logging.info(f"Reading indexing statistics from file: {filename}")
    tail_command = f"tail -n 10 {filename}"
    with subprocess.Popen(['/bin/bash', '-c', tail_command],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          cwd=os.getcwd(),
                          text=True
                          ) as proc:
        stdout, stderr = proc.communicate()
    lines = stdout.splitlines()
    for line in lines:
        if line.startswith("Final:"):
            indexing_statistics = line
    logging.info("\n {:s}".format(indexing_statistics))


if __name__ == "__main__":
    main()
