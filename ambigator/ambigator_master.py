#!/usr/bin/env python3

import shutil
import glob
import matplotlib.pyplot as plt
import logging
import functools
import sys
import argparse
import os
import subprocess
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # Use a non-interactive backend for matplotlib

log_filename = "ambigator_master.log"
log_level = logging.INFO


def parsing_arguments():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description='''Placeholder''',
        epilog='''Tamas Foldes - ESRF - 2025/../..''')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-i', '--inputfile', action='store', nargs=1,
                          type=str, required=True, dest="inpfile",
                          help="Input stream file")
    required.add_argument('-p', '--spacegroup', action='store', nargs=1,
                          type=str, required=True, dest="spacegroup",
                          help="Point group of the crystal")
    required.add_argument('-O', '--operator', action='store', nargs=1,
                          type=str, required=True, dest="operator",
                          help="Operator to be used for the analysis, for example: k,h,-l")
    optional.add_argument('-n', '--nchunks', action='store', default=6000,
                          type=int, dest="nchunks", nargs=1,
                          help="Number of chunks per slice file. The bias will be half of this value. The last slice might be smaller.")
    required.add_argument('-j', '--ncores', action='store', nargs=1,
                          type=int, required=True, dest="ncores",
                          help="Number of cores used by the ambigator")
    optional.add_argument('-r', '--niter', action='store', default=6,
                          type=int, dest="niter", nargs=1,
                          help="Number of ambigator iterations.")
    optional.add_argument('-o', '--output_file', action='store', default="processed.stream",
                          type=str, dest="output_file", nargs=1,
                          help="Name of the output stream file.")
    # optional.add_argument('-m', '--maxchunks', action='store', default=-1,
    #                       type=int, dest="max_total_chunks", nargs=1,
    #                       help="Maximum number of total chunks to \
    #                             process (None for no limit).")
    args = parser.parse_args()
    if isinstance(args.inpfile, list):
        args.inpfile = args.inpfile[0]
    if isinstance(args.spacegroup, list):
        args.spacegroup = args.spacegroup[0]
    if isinstance(args.operator, list):
        args.operator = args.operator[0]
    if isinstance(args.nchunks, list):
        args.nchunks = args.nchunks[0]
    if isinstance(args.ncores, list):
        args.ncores = args.ncores[0]
    if isinstance(args.niter, list):
        args.niter = args.niter[0]
    if isinstance(args.output_file, list):
        args.output_file = args.output_file[0]

    dataset = AmbigatorProcess(args)

    return dataset


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


@log_to_file(log_filename, log_level)
def split_stream_file(input_file, output_prefix, max_chunks=5000,
                      write_interval=200, max_total_chunks=None) -> list:
    """
    Splits a large text file into smaller parts while keeping chunks intact, 
    adding the header to each part, and logs time statistics. Writes data 
    incrementally every 'write_interval' chunks and shows progress.

    Parameters:
        input_file (str): Path to the input text file.
        output_prefix (str): Prefix for output files.
        max_chunks (int): Maximum number of chunks per output file.
        write_interval (int): Number of chunks after which the file is updated.
        max_total_chunks (int or None): Maximum number of total chunks to 
                                        process (None for no limit).
    Outputs:
        ofilenames (list of str) : List of output file names created.
        n_crystals (list of int) : Number of crystals in each output file.
    """
    logging.info(f"Splitting stream file: {input_file}")
    logging.info(f"Number of chunks per file: {max_chunks}")
    if max_total_chunks is not None:
        logging.info(
            f"Maximum number of chunks to process: {max_total_chunks}")

    ofilenames = []
    n_crystals = [0]

    with open(input_file, 'r', encoding='utf-8', buffering=1024 * 1024) as f:

        # Read and store header
        logging.debug("Reading header")
        header = []
        for line in f:
            header.append(line)
            if line.startswith("Indexing methods selected:"):
                break  # Last line of header found
        logging.debug("Header read complete")

        # Initialize variables for chunk collection
        chunk_count = 0
        total_chunk_count = 0
        file_count = 1
        output_file = "{:s}_{:>03d}.stream".format(output_prefix, file_count)

        # Buffer for chunks to be written
        chunks_buffer = []

        # Open the first output file and write the header
        logging.info(f"Opening output file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(header)
            ofilenames.append(output_file)
            logging.debug(f"Header added to {output_file}")

        # Read the file line by line
        for index, line in enumerate(f):
            if line.startswith("--- Begin crystal"):
                n_crystals[-1] += 1

            if line.startswith("----- Begin chunk -----"):

                # Stop if maximum total chunks reached
                if max_total_chunks and total_chunk_count >= max_total_chunks:
                    logging.info(
                        f"Maximum number of chunks reached.")
                    break

                # Write data every 'write_interval' chunks
                if chunk_count % write_interval == 0 and chunk_count > 0:
                    with open(output_file, 'a', encoding='utf-8') as out_f:
                        out_f.writelines(chunks_buffer)
                    chunks_buffer = []  # Clear chunk buffer after writing

                # If we reach max chunks per file, start a new file
                if chunk_count >= max_chunks:
                    logging.info(
                        f"Stream file completed: {output_file} | {total_chunk_count:>6d} chunks written")
                    file_count += 1
                    n_crystals.append(0)  # Reset crystal count for new file
                    output_file = "{:s}_{:>03d}.stream".format(
                        output_prefix, file_count)
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        logging.info(f"Opening output file: {output_file}")
                        logging.debug(f"Header added to {output_file}")
                        out_f.writelines(header)
                        ofilenames.append(output_file)
                    chunk_count = 0  # Reset chunk count for the new file

                chunk_count += 1
                total_chunk_count += 1

            chunks_buffer.append(line)

        # Write any remaining chunks at the end of the file
        if len(chunks_buffer) > 0:
            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.writelines(chunks_buffer)
            logging.info(
                f"Stream file completed: {output_file} | {total_chunk_count:>6d} chunks written")

    logging.info(
        f"Processing complete. {total_chunk_count} chunks processed across {file_count} files.")

    lines = ["\nFilename            N_crystals"]
    for filename, n_crystal in zip(ofilenames, n_crystals):
        lines.append(f"{filename}   {n_crystal:>6d}")
    logging.info("\n".join(lines))

    return ofilenames, n_crystals


@log_to_file(log_filename, log_level=log_level)
def combine_streams(inpfile1, inpfile2, outfile, buffer_size=200):

    if isinstance(inpfile2, str):
        inpfile2 = [inpfile2]
    N_files = len(inpfile2)+1
    logging.info(
        f"Combining stream files. \nFirst file: {inpfile1}\nTotal number of files {N_files}")

    with open(outfile, "w") as ofile:
        cnt = 0
        line_buffer = []
        with open(inpfile1) as inpfile:
            for line in inpfile:
                line_buffer.append(line)
                if len(line_buffer) % buffer_size == 0:
                    cnt += len(line_buffer)
                    ofile.writelines(line_buffer)
                    ofile.flush()
                    line_buffer = []
        if len(line_buffer) > 0:
            cnt += len(line_buffer)
            ofile.writelines(line_buffer)
            ofile.flush()
        logging.info(f"Finished writing {inpfile1} to {outfile}")

        for inpfile2_proc in inpfile2:
            line_buffer = []
            is_header = True
            cnt = 0
            with open(inpfile2_proc) as inpfile:
                for line in inpfile:
                    if is_header:
                        if line.startswith("Indexing methods selected:"):
                            is_header = False
                    else:
                        line_buffer.append(line)
                        if len(line_buffer) % buffer_size == 0:
                            cnt += len(line_buffer)
                            ofile.writelines(line_buffer)
                            ofile.flush()
                            line_buffer = []
                if len(line_buffer) > 0:
                    ofile.writelines(line_buffer)
                    ofile.flush()
                    cnt += len(line_buffer)
            logging.info(f"Finished writing {inpfile2_proc} to {outfile}")


@log_to_file(log_filename, log_level=log_level)
def run_command(command,
                out_filename="command_output.log",
                err_filename="command_error.log"):

    def get_formatted_command(text, num_spaces=4):
        formatted_text = re.sub(r'\s{2,}', '\n', text)
        lines = formatted_text.splitlines()
        indentation = " " * num_spaces
        return "\n".join([lines[0]] + [indentation + line for line in lines[1:]])

    formatted_command = get_formatted_command(
        command, num_spaces=8)
    logging.info(f"Running command:\n{formatted_command}")
    with open(out_filename, "w") as out, open(err_filename, "w") as err:
        with subprocess.Popen(['/bin/bash', '-c', command],
                              stdout=out,
                              stderr=err,
                              cwd=os.getcwd(),
                              text=True
                              ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logging.error("Command failed")
                quit()
            else:
                logging.info("Command finished successfully")


@log_to_file(log_filename, log_level=log_level)
def run_ambigator(dataset,
                  ambigator_command,
                  output_log="ambigator_output.log",
                  error_log="ambigator_error.log"):
    run_command(command=ambigator_command,
                out_filename=output_log,
                err_filename=error_log)

    dataset.logfiles.append(output_log)
    dataset.logfiles.append(error_log)

    log_lines = [f"\nFinal lines in {error_log}\n"]
    with open(error_log) as inpfile:
        lines = inpfile.readlines()
        cnt = len(lines)-1
        while not lines[cnt].startswith("Loaded"):
            cnt -= 1
            if cnt < 0:
                break
        lines = [line for line in lines[cnt:] if len(line) > 3]
        log_lines = log_lines+lines
    logging.info("".join(log_lines))


class AmbigatorProcess():

    def __init__(self, args):
        self.inpfile = args.inpfile
        self.spacegroup = args.spacegroup
        self.operator = args.operator
        self.nchunks = args.nchunks
        self.ncores = args.ncores
        self.niter = args.niter
        self.output_file = args.output_file
        self.logfiles = []
        self.image_files = []
        self.pointgroup = self.convert_space_group(self.spacegroup)[0]

        self.sample_stream = "sample.stream"
        self.bias_stream = "bias.stream"

        self.ambigator_bias = f"""ambigator {self.sample_stream} \
                                            -o {self.bias_stream} \
                                            -y {self.pointgroup} \
                                            --operator={self.operator} \
                                            -j {self.ncores} \
                                            --fg-graph=ambi_bias_graph.dat \
                                            --iterations={args.niter}"""

    def generate_command_for_slice(self, idx):
        processed_filename = f"slice_biased_ambi_{idx+1:03d}.stream"
        ambigator_biased_slice = f"""ambigator {self.biased_filenames[idx]} \
                                                -o {processed_filename} \
                                                -y {self.pointgroup} \
                                                --operator={self.operator} \
                                                -j {self.ncores} \
                                                --fg-graph={self.slice_graph_files[idx]} \
                                                --iterations={self.niter} \
                                                --start-assignments={self.slice_start_assignments[idx]} \
                                                --end-assignments={self.slice_end_assignments[idx]}"""
        return ambigator_biased_slice

    @staticmethod
    def convert_space_group(space_group):
        space_group_data = """1,1,P1            2,-1,P-1          3,2,P2            4,2,P21           5,2,C2
                              6,m,Pm            7,m,Pc            8,m,Cm            9,m,Cc            10,2/m,P2/m
                              11,2/m,P21/m      12,2/m,C2/m       13,2/m,P2/c       14,2/m,P21/c      15,2/m,C2/c
                              16,222,P222       17,222,P2221      18,222,P21212     19,222,P212121    20,222,C2221
                              21,222,C222       22,222,F222       23,222,I222       24,222,I212121    25,mm2,Pmm2
                              26,mm2,Pmc21      27,mm2,Pcc2       28,mm2,Pma2       29,mm2,Pca21      30,mm2,Pnc2
                              31,mm2,Pmn21      32,mm2,Pba2       33,mm2,Pna21      34,mm2,Pnn2       35,mm2,Cmm2
                              36,mm2,Cmc21      37,mm2,Ccc2       38,mm2,Amm2       39,mm2,Abm2       40,mm2,Ama2
                              41,mm2,Aba2       42,mmm,Fmm2       43,mmm,Fdd2       44,mmm,Imm2       45,mmm,Iba2
                              46,mmm,Ima2       47,mmm,Pmmm       48,mmm,Pnnn       49,mmm,Pccm       50,mmm,Pban
                              51,mmm,Pmma       52,mmm,Pnna       53,mmm,Pmna       54,mmm,Pcca       55,mmm,Pbam
                              56,mmm,Pccn       57,mmm,Pbcm       58,mmm,Pnnm       59,mmm,Pmmn       60,mmm,Pbcn
                              61,mmm,Pbca       62,mmm,Pnma       63,mmm,Cmcm       64,mmm,Cmca       65,mmm,Cmmm
                              66,mmm,Cccm       67,mmm,Cmma       68,mmm,Ccca       69,mmm,Fmmm       70,mmm,Fddd
                              71,mmm,Immm       72,mmm,Ibam       73,mmm,Ibca       74,mmm,Imma       75,4,P4
                              76,4,P41          77,4,P42          78,4,P43          79,4,I4           80,4,I41
                              81,-4,P-4         82,-4,I-4         83,4/m,P4/m       84,4/m,P42/m      85,4/m,P4/n
                              86,4/m,P42/n      87,4/m,I4/m       88,4/m,I41/a      89,422,P422       90,422,P4212
                              91,422,P4122      92,422,P41212     93,422,P4222      94,422,P42212     95,422,P4322
                              96,422,P43212     97,422,I422       98,422,I4122      99,4mm,P4mm       100,4mm,P4bm
                              101,4mm,P42cm     102,4mm,P42nm     103,4mm,P4cc      104,4mm,P4nc      105,4mm,P42mc
                              106,4mm,P42bc     107,4mm,I4mm      108,4mm,I4cm      109,4mm,I41md     110,4mm,I41cd
                              111,-42m,P-42m    112,-42m,P-42c    113,-42m,P-421m   114,-42m,P-421c   115,-42m,P-4m2
                              116,-42m,P-4c2    117,-42m,P-4b2    118,-42m,P-4n2    119,-42m,I-4m2    120,-42m,I-4c2
                              121,-42m,I-42m    122,-42m,I-42d    123,4/mmm,P4/mmm  124,4/mmm,P4/mcc  125,4/mmm,P4/nbm
                              126,4/mmm,P4/nnc  127,4/mmm,P4/mbm  128,4/mmm,P4/mnc  129,4/mmm,P4/nmm  130,4/mmm,P4/ncc
                              131,4/mmm,P42/mmc 132,4/mmm,P42/mcm 133,4/mmm,P42/nbc 134,4/mmm,P42/nnm 135,4/mmm,P42/mbc
                              136,4/mmm,P42/mnm 137,4/mmm,P42/nmc 138,4/mmm,P42/ncm 139,4/mmm,I4/mmm  140,4/mmm,I4/mcm
                              141,4/mmm,I41/amd 142,4/mmm,I41/acd 143,3,P3          144,3,P31         145,3,P32
                              146,3,R3          147,-3,P-3        148,-3,R-3        149,32,P312       150,32,P321
                              151,32,P3112      152,32,P3121      153,32,P3212      154,32,P3221      155,32,R32
                              156,3m,P3m1       157,3m,P31m       158,3m,P3c1       159,3m,P31c       160,3m,R3m
                              161,3m,R3c        162,-3m,P-31m     163,-3m,P-31c     164,-3m,P-3m1     165,-3m,P-3c1
                              166,-3m,R-3m      167,-3m,R-3c      168,6,P6          169,6,P61         170,6,P65
                              171,6,P62         172,6,P64         173,6,P63         174,-6,P-6        175,6/m,P6/m
                              176,6/m,P63/m     177,622,P622      178,622,P6122     179,622,P6522     180,622,P6222
                              181,622,P6422     182,622,P6322     183,6mm,P6mm      184,6mm,P6cc      185,6mm,P63cm
                              186,6mm,P63mc     187,-6m2,P-6m2    188,-6m2,P-6c2    189,-6m2,P-62m    190,-6m2,P-62c
                              191,6/mmm,P6/mmm  192,6/mmm,P6/mcc  193,6/mmm,P63/mcm 194,6/mmm,P63/mmc 195,23,P23
                              196,23,F23        197,23,I23        198,23,P213       199,23,I213       200,m-3,Pm-3
                              201,m-3,Pn-3      202,m-3,Fm-3      203,m-3,Fd-3      204,m-3,Im-3      205,m-3,Pa-3
                              206,m-3,Ia-3      207,432,P432      208,432,P4232     209,432,F432      210,432,F4132
                              211,432,I432      212,432,P4332     213,432,P4132     214,432,I4132     215,-43m,P-43m
                              216,-43m,F-43m    217,-43m,I-43m    218,-43m,P-43n    219,-43m,F-43c    220,-43m,I-43d
                              221,m-3m,Pm-3m    222,m-3m,Pn-3n    223,m-3m,Pm-3n    224,m-3m,Pn-3m    225,m-3m,Fm-3m
                              226,m-3m,Fm-3c    227,m-3m,Fd-3m    228,m-3m,Fd-3c    229,m-3m,Im-3m    230,m-3m,Ia-3d"""
        space_group_data = space_group_data.replace("\n", " ").split()
        space_group_data = np.array([line.split(",")
                                    for line in space_group_data])

        if space_group not in space_group_data[:, 2]:
            logging.error(
                f"Space group {space_group} not found in the database.")
            quit()
        else:
            pointgroup = space_group_data[space_group_data[:, 2]
                                          == space_group][0, 1]
            spacegroup_number = space_group_data[space_group_data[:, 2]
                                                 == space_group][0, 0]
            logging.info(
                f"Space group {space_group} converted to point group: {pointgroup}, space group number: {spacegroup_number}")
            return pointgroup, spacegroup_number


@log_to_file(log_filename, log_level=log_level)
def generate_ambigator_starting_assignments(N, N_bias, ofilename, seed=42):

    def write_assignments(ofilename, assignments):
        with open(ofilename, "w") as ofile:
            for assign in assignments:
                if assign:
                    ofile.write("1\n")
                else:
                    ofile.write("0\n")

    np.random.seed(seed)
    assignments = np.full(shape=N, fill_value=False).astype(bool)
    to_be_changed_to_True = np.random.choice(a=np.array(list(range(int((N-N_bias))))),
                                             size=int((N-N_bias)//2),
                                             replace=False)
    to_be_changed_to_True = np.sort(to_be_changed_to_True)
    assignments[to_be_changed_to_True+N_bias] = True
    write_assignments(ofilename, assignments)
    logging.info(
        f"Starting assignments written to {ofilename} with bias size {N_bias} ({N_bias/N:.2%}) of the total number of crystals {N}")


@log_to_file(log_filename, log_level)
def plot_data(datafile, NITER, pngfile="correlation_python.png"):

    # Load data
    data = np.loadtxt(datafile)
    x = np.arange(len(data))  # Implicit x-values
    y1 = data[:, 0]           # First data column
    y2 = data[:, 1]           # Second data column

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot x-axis zero line
    plt.axhline(0, color="black", ls="-", lw=2)

    # Plot points
    kwargs = {"s": 5, "marker": '|', "alpha": 0.5, "lw": 1.5}
    plt.scatter(x, y2, color="#0088FF", **kwargs)
    plt.scatter(x, y1, color="orange",  **kwargs)

    kwargs2 = {"s": 50, "marker": 's', "alpha": 1.0, "lw": 1.0}
    plt.scatter(np.nan, np.nan, color="orange", label="f", **kwargs2)
    plt.scatter(np.nan, np.nan, color="#0088FF", label="g", **kwargs2)

    # Random filter equivalent (like rand(0) > 0.5), only for visualization
    random_filter = np.random.rand(len(x)) > 0.5
    plt.scatter(x[random_filter], y1[random_filter],
                color="orange", s=5, marker='|', alpha=0.5, lw=1.5)

    npoints = 50
    smoothed_x = np.linspace(x[0], x[-1], npoints)
    bins = np.digitize(x, 0.5*(smoothed_x[1:]+smoothed_x[:-1]))
    smoothed_col1, smoothed_col2 = [], []
    minvals, maxvals = [], []
    for b in np.unique(bins):
        smoothed_col1.append(np.average(y1[bins == b]))
        smoothed_col2.append(np.average(y2[bins == b]))
        minvals.append(np.average(np.minimum(y1[bins == b], y2[bins == b])))
        maxvals.append(np.average(np.maximum(y1[bins == b], y2[bins == b])))
    smoothed_col1 = np.array(smoothed_col1)
    smoothed_col2 = np.array(smoothed_col2)
    minvals = np.array(minvals)
    maxvals = np.array(maxvals)

    plt.plot(smoothed_x, maxvals,       color="black", lw=2.0)
    plt.plot(smoothed_x, minvals,       color="black", lw=2.0)
    plt.plot(smoothed_x, smoothed_col1, color="red", lw=2.0, label="avg(f)")
    plt.plot(smoothed_x, smoothed_col2, color="blue", lw=2.0, label="avg(g)")
    plt.plot(np.nan, np.nan, color='k', lw=1.0, label="avg(f)-avg(g)")

    # Labels and axis settings
    plt.xlabel("Number of crystals")
    plt.ylabel("Correlation")
    plt.xlim(x[0], x[-1])

    average = np.average(data)
    std = np.std(data)
    plt.ylim(average-3.5*std, average+3.5*std)

    # Plot reference vertical line at the end of the first iteration
    plt.axvline(len(x)/NITER, color="black", ls=(0, (1.5, 2.0)),
                lw=1.0, dash_capstyle='round')

    # Legend
    plt.legend(loc="lower right", framealpha=1.0)

    # Secondary x-axis (Number of passes)
    ax2 = ax.twiny()
    ax2.set_xlim(0, NITER)
    ax2.set_xlabel("Number of passes over all crystals")

    ax3 = ax.twinx()
    ax3.plot(smoothed_x, smoothed_col1-smoothed_col2, color='k', lw=1.0)

    plt.tight_layout()

    # Save figure
    plt.savefig(pngfile, dpi=300)
    plt.close()


@log_to_file(log_filename, log_level)
def compare_flips(dataset):

    esrf_blue = (19/255, 37/255, 119/255, 1.0)
    esrf_orange = (237/255, 119/255,  3/255, 1.0)

    def read_data(filename):
        with open(filename) as inpfile:
            data = inpfile.readlines()
        data = np.array([val.strip() == "1" for val in data]).astype(bool)
        return data

    def comparison(st, en):
        temp_st = np.zeros(shape=len(st))
        temp_st[st == True] = 1
        temp_en = np.zeros(shape=len(en))
        temp_en[en == True] = 1
        my_rho = np.corrcoef(temp_st, temp_en)
        return my_rho[0, 1]

    def compare_boolean_arrays(arr1: np.ndarray, arr2: np.ndarray) -> float:
        if arr1.size != arr2.size:
            raise ValueError("Arrays must have the same number of elements")
        same = np.sum(arr1 == arr2)
        different = np.sum(arr1 != arr2)
        return (same - different) / arr1.size

    Nbias = dataset.n_crystals_bias
    start_filenames = dataset.slice_start_assignments
    end_filenames = dataset.slice_end_assignments

    start_data = []
    for filename in start_filenames:
        start_data.append(read_data(filename))

    end_data = []
    for filename in end_filenames:
        end_data.append(read_data(filename))

    fig, ax = plt.subplots(figsize=(8, 4))
    cnt = 0
    for st, en in zip(start_data, end_data):
        cnt += 1
        xs = np.arange(len(st))+1
        ys = np.full(shape=len(st), fill_value=cnt)
        plt.scatter(-2.0, cnt+0.25, c="w", marker=".")
        plt.scatter(xs[en == False], ys[en == False],
                    color=esrf_orange, marker='.', lw=0.1, s=1)
        plt.scatter(xs[en == True], ys[en == True],
                    color=esrf_blue, marker='|', lw=0.1, s=80)
        Nflipped_bias = (en[:Nbias] == True).sum()
        Nflipped_data = (en[Nbias:] == True).sum()
        plt.text(0, cnt+0.2,
                 "{:>8.1f}%".format(Nflipped_bias/Nbias*100), ha='left', va='bottom')
        plt.text(Nbias, cnt+0.2,
                 "{:>8.1f}%".format(Nflipped_data/(len(en)-Nbias)*100), ha='left', va='bottom')
    plt.axvline(Nbias, color='k', ls='--', lw=1.0)
    bottom, top = plt.ylim()
    plt.yticks(list(range(-1, len(start_data)+2, 1)))
    plt.ylim(bottom, top)
    plt.tight_layout()
    plt.savefig("ribbon_comparison.png", dpi=500)
    plt.close()

    correl = np.zeros(shape=(len(end_data), len(end_data)))
    for i in range(len(end_data)):
        correl[i, i] = 1.0
    for i in range(0, len(end_data)):
        for j in range(i+1, len(end_data)):
            c = compare_boolean_arrays(
                end_data[i][:Nbias], end_data[j][:Nbias])
            correl[i, j] = c
            correl[j, i] = c

    fig, ax = plt.subplots(figsize=(5, 5))

    xs = np.arange(0, 10, 1)
    plt.xticks(ticks=xs+0.5, labels=["{:.0f}".format(x+1) for x in xs])
    plt.yticks(ticks=xs+0.5, labels=["{:.0f}".format(x+1) for x in xs])
    c = ax.imshow(correl, cmap='bwr_r', vmin=0.1, vmax=1.2,
                  extent=[0, len(end_data), 0, len(end_data)],
                  interpolation='none', origin='lower', aspect='equal')
    for i in range(len(end_data)):
        plt.text(
            i+0.5, i+0.5, "{:.3f}".format(correl[i, i]), ha='center', va='center', fontsize=11)
    for i in range(0, len(end_data)):
        for j in range(i+1, len(end_data)):
            plt.text(
                i+0.5, j+0.5, "{:.2f}".format(correl[i, j]), ha='center', va='center', fontsize=11)
            plt.text(
                j+0.5, i+0.5, "{:.2f}".format(correl[i, j]), ha='center', va='center', fontsize=11)
    plt.tight_layout()
    plt.savefig("correl_matrix.png", dpi=300)


@log_to_file(log_filename, log_level)
def delete_first_chunks(input_file, output_file, deleted_chunks,
                        write_interval=100):

    with open(input_file, 'r', encoding='utf-8') as f:
        header = []
        for line in f:
            header.append(line)
            if line.startswith("Indexing methods selected:"):
                break  # Last line of header found
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(header)
            cnt = 0
            chunk_buffer = []
            for line in f:
                if line.startswith("----- Begin chunk -----"):
                    cnt += 1

                if cnt > deleted_chunks:
                    chunk_buffer.append(line)

                    if (cnt-deleted_chunks) % write_interval == 0:
                        out_f.writelines(chunk_buffer)
                        chunk_buffer = []
            if len(chunk_buffer) > 0 and cnt > deleted_chunks:
                out_f.writelines(chunk_buffer)


@log_to_file(log_filename, log_level)
def move_files_to_folder(foldername, filename):
    if not os.path.exists(foldername):
        try:
            os.makedirs(foldername)
            logging.info(f"Created folder: {foldername}")
        except OSError as e:
            logging.error(f"Error creating folder {foldername}: {e}")
            return
    elif not os.path.isdir(foldername):
        logging.error(f"Error: {foldername} exists but is not a directory")
        return

    current_dir = os.getcwd()
    source_path = os.path.join(current_dir, filename)
    dest_path = os.path.join(current_dir, foldername, filename)
    try:
        shutil.move(source_path, dest_path)
        logging.info(f"Moved: {filename} -> {foldername}/")
    except shutil.Error as e:
        logging.error(f"Error moving {filename}: {e}")
    except OSError as e:
        logging.error(f"Error moving {filename}: {e}")


@log_to_file(log_filename, log_level)
def main():
    dataset = parsing_arguments()

    # 1. cut sample slice
    _, N = split_stream_file(input_file=dataset.inpfile,
                             output_prefix="sample",
                             max_chunks=int(dataset.nchunks//2),
                             max_total_chunks=int(dataset.nchunks//2))
    os.rename('sample_001.stream', 'sample.stream')
    dataset.n_crystals_bias = N[0]

    # 2. run ambigator on sample -> bias.stream
    output_log = "ambigator_bias.log"
    error_log = "ambigator_bias_error.log"
    run_ambigator(dataset=dataset,
                  ambigator_command=dataset.ambigator_bias,
                  output_log=output_log,
                  error_log=error_log)
    dataset.image_files.append("ambi_bias_graph.dat")
    dataset.logfiles += [output_log, error_log]

    # 3. cut the original stream to slices
    filenames, n_crystals = split_stream_file(input_file=dataset.inpfile,
                                              output_prefix="slice",
                                              max_chunks=dataset.nchunks)
    dataset.slices = filenames
    dataset.n_crystals = n_crystals

    # 4. merge the bias.stream with the slices
    dataset.biased_filenames = []
    for i, filename in enumerate(dataset.slices, 1):
        biased_filename = f"slice_biased_{i:03d}.stream"
        combine_streams(inpfile1=dataset.bias_stream,
                        inpfile2=filename,
                        outfile=biased_filename)
        dataset.biased_filenames.append(biased_filename)

    # 5. generate the ambigator starting assignments
    dataset.slice_start_assignments = []
    dataset.slice_end_assignments = []
    dataset.slice_graph_files = []
    for i, filename in enumerate(dataset.biased_filenames):
        biased_assignment_filename = f"slice_biased_{i+1:03d}_assign.dat"
        generate_ambigator_starting_assignments(N=dataset.n_crystals[i]+dataset.n_crystals_bias,
                                                N_bias=dataset.n_crystals_bias,
                                                ofilename=biased_assignment_filename,
                                                seed=42+i)
        dataset.slice_start_assignments.append(biased_assignment_filename)
        biased_end_filename = f"slice_biased_{i+1:03d}_end.dat"
        dataset.slice_end_assignments.append(biased_end_filename)
        slice_graph_filename = f"slice_biased_{i+1:03d}_graph.dat"
        dataset.slice_graph_files.append(slice_graph_filename)

    # 6. run ambigator on the biased slices
    for i in range(len(dataset.biased_filenames)):
        ambigator_command = dataset.generate_command_for_slice(i)
        output_log = f"ambigator_biased_slice_{i+1:03d}.log"
        error_log = f"ambigator_biased_slice_{i+1:03d}_error.log"
        run_ambigator(dataset=dataset,
                      ambigator_command=ambigator_command,
                      output_log=output_log,
                      error_log=error_log)
        dataset.image_files.append(dataset.slice_graph_files[i])
        dataset.logfiles += [output_log, error_log]

    # 7. create plots about the ambigator
    plot_data(datafile="ambi_bias_graph.dat",
              NITER=dataset.niter,
              pngfile="correlation_bias.png")
    dataset.image_files.append("correlation_bias.png")
    for i, filename in enumerate(dataset.slice_graph_files):
        pngfile = f"correlation_slice_{i+1:03d}.png"
        plot_data(datafile=filename,
                  NITER=dataset.niter,
                  pngfile=pngfile)
        dataset.image_files.append(pngfile)
    compare_flips(dataset=dataset)

    # 8. delete the bias part of the slices
    dataset.cleaned_filenames = []
    for i, filename in enumerate(dataset.biased_filenames):
        cleaned_filename = f"slice_cleaned_{i+1:03d}.stream"
        delete_first_chunks(input_file=filename,
                            output_file=cleaned_filename,
                            deleted_chunks=int(dataset.nchunks//2))
        dataset.cleaned_filenames.append(cleaned_filename)

    # 9. merge the slices to final processed.stream
    combine_streams(inpfile1=dataset.cleaned_filenames[0],
                    inpfile2=dataset.cleaned_filenames[1:],
                    outfile=dataset.output_file)

    # 10. clean up
    files_to_be_deleted = ["sample.stream", "bias.stream"]
    for filename in dataset.cleaned_filenames:
        files_to_be_deleted.append(filename)
    for filename in dataset.biased_filenames:
        files_to_be_deleted.append(filename)
    for filename in dataset.slices:
        files_to_be_deleted.append(filename)
    for filename in files_to_be_deleted:
        if os.path.exists(filename):
            os.remove(filename)
            logging.info(f"Deleted {filename}")
        else:
            logging.warning(f"{filename} not found, skipping deletion")
    for filename in dataset.image_files:
        if os.path.exists(filename):
            move_files_to_folder("images", filename)
            logging.info(f"Moved {filename} to images folder")
    for filename in dataset.logfiles:
        if os.path.exists(filename):
            move_files_to_folder("logs", filename)
            logging.info(f"Moved {filename} to logs folder")

if __name__ == "__main__":
    main()
