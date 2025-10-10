#!/usr/bin/env python3

import sys
import logging
import functools
import os
import numpy as np
import subprocess
import glob
import re
import shutil

log_filename = "process_stream.log"
log_level = logging.INFO


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


def log_to_file(log_filename, log_level=logging.INFO):
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
    space_group_data = np.array([line.split(",") for line in space_group_data])

    if space_group not in space_group_data[:, 2]:
        logging.error(f"Space group {space_group} not found in the database.")
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
def find_files_by_pattern(pattern):
    logging.debug(f"Searching for files with pattern:\n {pattern}")
    filenames = glob.glob(pattern)
    filenames.sort()
    return filenames


@log_to_file(log_filename, log_level=log_level)
def extract_unit_cell(cell_file):
    with open(cell_file, "r") as cellfile:
        cell = cellfile.readlines()
        cell = [line for line in cell if len(line) > 2]
        parameters = ["a", "b", "c", "al", "be", "ga"]
        cell = [line.split()[2]
                for line in cell if line.split()[0] in parameters]
        logging.info(f"Unit cell extracted from file {cell_file}")
        logging.debug(f"Unit cell parameters: {parameters}")
        logging.info(f"Unit cell parameters: {cell}")
        return cell


@log_to_file(log_filename, log_level=logging.INFO)
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


@log_to_file(log_filename, log_level=logging.INFO)
def run_xscale(dataset):
    # Print the header

    xscale_header = f"""!FORMAT=XDS_ASCII   MERGE=TRUE   FRIEDEL'S_LAW={dataset.Friedels_law}
                        !SPACE_GROUP_NUMBER={dataset.spacegroup_number}
                        !UNIT_CELL_CONSTANTS={" ".join(dataset.unit_cell)}
                        !NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=5
                        !X-RAY_WAVELENGTH={dataset.wavelength}
                        !ITEM_H=1
                        !ITEM_K=2
                        !ITEM_L=3
                        !ITEM_IOBS=4
                        !ITEM_SIGMA(IOBS)=5
                        !END_OF_HEADER"""

    xds_hkl_file = "xds.hkl"

    with open(xds_hkl_file, "w") as xscale_input:
        for line in xscale_header.splitlines():
            xscale_input.write(f"{line.lstrip()}\n")

        # Define the regex pattern with named groups
        pattern = re.compile(
            r'^\s+'                    # one or more whitespace characters
            r'(?P<h>[-0-9]+)'          # h: one or more digits or negative sign
            r'\s+'                     # one or more whitespace characters
            r'(?P<k>[-0-9]+)'          # k: one or more digits or negative sign
            r'\s+'                     # one or more whitespace characters
            r'(?P<l>[-0-9]+)'          # l: one or more digits or negative sign
            r'\s+'                     # one or more whitespace characters
            r'(?P<intensity>[0-9.-]+)'  # intensity: float, (neg. sign)
            r'\s+'                     # one or more whitespace characters
            r'(?P<phase>[-+])'         # phase: a single "-" (ignored)
            r'\s+'                     # one or more whitespace characters
            r'(?P<sigma>[0-9.-]+)'     # sigma: float, (neg. sign)
            r'\s+'                     # one or more whitespace characters
            r'(?P<nmeas>[0-9]+)'       # nmeas: one or more digits (ignored)
            r'$'                       # End of the line
        )

        # Process the file
        with open(dataset.hkl_file, 'r') as fh:
            for line in fh:

                # Use the pattern to match the line
                match = pattern.match(line)

                if match:
                    h = int(match.group('h'))
                    k = int(match.group('k'))
                    l = int(match.group('l'))
                    intensity = float(match.group('intensity'))
                    # Using named group 'sigma'
                    sigma = float(match.group('sigma'))

                    xscale_input.write(
                        f"{h:6d} {k:6d} {l:5d} {intensity:9.2f} {sigma:9.2f}\n")
                else:
                    # print(f"Unrecognised: '{line}'", file=sys.stderr)
                    pass

        xscale_input.write("!END_OF_DATA\n")

    with open("XDSCONV.INP", "w") as xdsconv_input:
        lines = f"""OUTPUT_FILE=ccp4.hkl CCP4_I+F
                    INPUT_FILE=xds.hkl
                    GENERATE_FRACTION_OF_TEST_REFLECTIONS=0.05
                    INCLUDE_RESOLUTION_RANGE={dataset.lowres} {dataset.highres}"""
        lines = [f"{line.lstrip()}\n" for line in lines.splitlines()]
        xdsconv_input.writelines(lines)

    with open("xscale_xdsconv_out.log", "w") as out, open("xscale_xdsconv_err.log", "w") as err:
        with subprocess.Popen(['/bin/bash', '-c', "xdsconv </dev/null"],
                              stdout=out,
                              stderr=err,
                              cwd=os.getcwd(),
                              text=True
                              ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logging.error("xdsconv failed")
                quit()
            else:
                logging.info("xdsconv finished successfully")

    with open("xscale_f2mtz_out.log", "w") as out, open("xscale_f2mtz_err.log", "w") as err:
        with subprocess.Popen(['/bin/bash', '-c', f"f2mtz HKLOUT temp.mtz <F2MTZ.INP"],
                              stdout=out,
                              stderr=err,
                              cwd=os.getcwd(),
                              text=True
                              ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logging.error("f2mtz failed")
                quit()
            else:
                logging.info("f2mtz finished successfully")

    with open("xscale_cad_out.log", "w") as out, open("xscale_cad_err.log", "w") as err:
        cad_command = f"""cad HKLIN1 temp.mtz"""
        cad_command = cad_command + \
            f""" HKLOUT {dataset.project_name}_xscale.mtz"""
        cad_command = cad_command + f""" <<EOF_CAD\nLABIN FILE 1 ALL\nEND\nEOF_CAD"""
        with subprocess.Popen(['/bin/bash', '-c', cad_command],
                              stdout=out,
                              stderr=err,
                              cwd=os.getcwd(),
                              text=True
                              ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logging.error("cad failed")
                quit()
            else:
                logging.info("cad finished successfully")

    dataset.logfiles += find_files_by_pattern("xscale*.log")
    dataset.tempfiles += find_files_by_pattern("*.hk*")
    dataset.tempfiles += find_files_by_pattern("*.xml")
    dataset.tempfiles += find_files_by_pattern("*.json")
    dataset.tempfiles += find_files_by_pattern("XDSCONV.*")
    dataset.tempfiles += find_files_by_pattern("F2MTZ.INP")
    dataset.tempfiles += find_files_by_pattern("temp.mtz")


@log_to_file(log_filename, log_level=log_level)
def run_partialator(dataset):
    partialator_command = f"""partialator -i {dataset.stream_file} \
                                          --model=unity            \
                                          -j 10                     \
                                          -o {dataset.hkl_file}    \
                                          -y {dataset.pointgroup}  \
                                          --no-pr                  \
                                          --no-logs                \
                                          --iterations=3"""

    run_command(command=partialator_command,
                out_filename="partialator_output.log",
                err_filename="partialator_error.log")

    dataset.logfiles += find_files_by_pattern("partialator*.log")


@log_to_file(log_filename, log_level=log_level)
def run_check_hkl(dataset):
    check_hkl_command = f"""check_hkl -y {dataset.pointgroup}     \
                                      -p {dataset.cell_file}      \
                                      --zero-negs                 \
                                      --nshells=30                \
                                      --shell-file=shells.dat     \
                                      --highres={dataset.highres} \
                                      {dataset.hkl_file}"""

    run_command(command=check_hkl_command,
                out_filename="check_hkl_output.log",
                err_filename="check_hkl_error.log")

    dataset.logfiles += find_files_by_pattern("check_hkl*.log")
    dataset.tempfiles.append("shells.dat")


@log_to_file(log_filename, log_level=log_level)
def run_import_serial(dataset):
    import_serial_command = f"""import_serial --hklin {dataset.hkl_file} \
                                              --half-dataset {dataset.hkl_file[:-4]}.hkl1 {dataset.hkl_file[:-4]}.hkl2  \
                                              --wavelength {dataset.wavelength} \
                                              --spacegroup {dataset.spacegroup} \
                                              --cellfile {dataset.cell_file} \
                                              --project {dataset.project_name} \
                                              --lowres {dataset.lowres} \
                                              --highres {dataset.highres} \
                                              --nshells 30"""
    run_command(command=import_serial_command,
                out_filename="import_serial_output.log",
                err_filename="import_serial_error.log")

    with open("import_serial_output.log") as inpfile:
        lines = inpfile.readlines()
        idx = 0
        while "DATA STATISTICS" not in lines[idx]:
            idx += 1
        lines = lines[idx-1:-2]
        logging.info("".join(lines))

    dataset.logfiles += find_files_by_pattern("import_serial*.log")


@log_to_file(log_filename, log_level=log_level)
def run_compare_hkl(dataset):
    compare_hkl_CCstar_command = f"""compare_hkl -y {dataset.pointgroup}      \
                                                 -p {dataset.cell_file}       \
                                                 --ignore-negs                \
                                                 --nshells=30                 \
                                                 --shell-file=ccstar.dat      \
                                                 --fom=CCstar                 \
                                                 {dataset.hkl_file[:-4]}.hkl1 \
                                                 --highres={dataset.highres}  \
                                                 {dataset.hkl_file[:-4]}.hkl2"""

    run_command(command=compare_hkl_CCstar_command,
                out_filename="compare_hkl_CCstar_output.log",
                err_filename="compare_hkl_CCstar_error.log")

    compare_hkl_Rsplit_command = f"""compare_hkl -y {dataset.pointgroup}      \
                                                 -p {dataset.cell_file}       \
                                                 --ignore-negs                \
                                                 --nshells=30                 \
                                                 --shell-file=Rsplit.dat      \
                                                 --fom=Rsplit                 \
                                                 {dataset.hkl_file[:-4]}.hkl1 \
                                                 --highres={dataset.highres}  \
                                                 {dataset.hkl_file[:-4]}.hkl2"""

    run_command(command=compare_hkl_Rsplit_command,
                out_filename="compare_hkl_Rsplit_output.log",
                err_filename="compare_hkl_Rsplit_error.log")

    dataset.logfiles += find_files_by_pattern("compare_hkl_*.log")
    dataset.tempfiles.append("ccstar.dat")
    dataset.tempfiles.append("Rsplit.dat")


class Dataset():
    def __init__(self, project_name, stream_file, cell_file, highres, spacegroup, Friedels_law):
        self.project_name = project_name
        self.hkl_file = "merged_partialator.hkl"
        self.stream_file = stream_file
        self.cell_file = cell_file
        self.lowres = 100
        self.highres = highres
        self.wavelength = 1.072  # wavelength at ID29
        self.spacegroup = spacegroup
        self.Friedels_law = Friedels_law
        self.logfiles = []
        self.tempfiles = []
        self.extract_unit_cell()

    def convert_space_group(self, pointgroup=None, spacegroup_number=None):
        PG, SGN = convert_space_group(self.spacegroup)
        if pointgroup is None:
            self.pointgroup = PG
        if spacegroup_number is None:
            self.spacegroup_number = SGN

    def extract_unit_cell(self):
        self.unit_cell = extract_unit_cell(self.cell_file)


@log_to_file(log_filename, log_level=log_level)
def run_ctruncate(dataset):
    ctruncate_command = f"""ctruncate -mtzin {dataset.project_name}_xscale.mtz    \
                                      -mtzout {dataset.project_name}_ctruncate.mtz    \
                                      -colano \"/*/*/[I(+),SIGI(+),I(-),SIGI(-)]\" \
                                      -freein \"/*/*/[FreeRflag]\""""

    run_command(command=ctruncate_command,
                out_filename="ctruncate_output.log",
                err_filename="ctruncate_error.log")
    dataset.logfiles += find_files_by_pattern("ctruncate*.log")


def move_files_to_folder(foldername, regex_pattern):
    # Check if the folder exists; create it if it doesn't
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

    # Find files matching the regex pattern
    files = glob.glob(regex_pattern)
    if not files:
        print(f"No files found matching pattern: {regex_pattern}")
        return
    current_dir = os.getcwd()
    for filename in files:
        source_path = os.path.join(current_dir, filename)
        dest_path = os.path.join(current_dir, foldername, filename)
        try:
            shutil.move(source_path, dest_path)
            logging.info(f"Moved: {filename} -> {foldername}/")
        except shutil.Error as e:
            logging.error(f"Error moving {filename}: {e}")
        except OSError as e:
            logging.error(f"Error moving {filename}: {e}")


@log_to_file(log_filename, log_level=log_level)
def clean_up_folder(dataset):
    for filename in list(set(dataset.logfiles)):
        move_files_to_folder("logs", filename)

    for filename in list(set(dataset.tempfiles)):
        move_files_to_folder("tempfiles", filename)


if __name__ == "__main__":
    logging.getLogger(__name__).setLevel(log_level)

    dataset = Dataset(project_name="MYO-int32",
                      stream_file=find_files_by_pattern("./*.stream")[0],
                      cell_file="../myoglobin_full.cell",
                      highres=1.0,
                      spacegroup="P212121",
                      Friedels_law="FALSE")
    dataset.convert_space_group()

    run_partialator(dataset)

    run_check_hkl(dataset)

    run_compare_hkl(dataset)

    run_import_serial(dataset)

    run_xscale(dataset)

    run_ctruncate(dataset)

    clean_up_folder(dataset)

    logging.info("Stream file processed.")
