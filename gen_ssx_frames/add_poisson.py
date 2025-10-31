import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import h5py as h5
import hdf5plugin
hdf5plugin.register()


def main():
    args = parse_args()

    logger = setup_logging(
        log_path=args.logfile,
        log_level=logging.INFO,
        overwrite_log=True,
    )

    logger.info(f"Adding Poisson noise to file {args.input}")
    logger.info(f"Noise level (lambda): {args.noise}")
    logger.info(f"Noisy data will be saved in {args.output}")

    add_poission_noise(
        inputfile=args.input,
        outputfile=args.output,
        lam=args.noise,
        seed_start=args.seed_start,
        chunksize=args.chunksize,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=(
            "Process HDF5 (H5) data files to add Poisson noise. "
        ),
        epilog=(
            "Example usage:\n"
            f"  python {sys.argv[0]} -i input.h5 -o output.h5 -n 2 -c 200 --force\n\n"
            "Tips:\n"
            "  • Use --force to overwrite an existing output file.\n"
            "  • Adjust --chunksize for performance tuning.\n"
            "  • The logfile records processing details and errors.\n"
            "==========================================="
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input HDF5 (.h5) file to be processed."
    )

    parser.add_argument(
        "-n", "--noise",
        type=int,
        default=1,
        help="Expected noise level (lambda). Controls random variation. (default: %(default)s)"
    )

    parser.add_argument(
        "-c", "--chunksize",
        type=int,
        default=100,
        help="Number of frames to process per chunk. Must evenly divide the total number of frames. (default: %(default)s)"
    )

    parser.add_argument(
        "-s", "--seed_start",
        type=int,
        default=42,
        help="Random seed for reproducibility. (default: %(default)s)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="testdata.h5",
        help="Path to the output HDF5 file to be created or overwritten. (default: %(default)s)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of the output file if it already exists. (default: False)"
    )

    parser.add_argument(
        "-l", "--logfile",
        type=str,
        default="h5_generation.log",
        help="Filename for logging processing information. (default: %(default)s)"
    )

    args = parser.parse_args()

    # --- Validation ---
    errors = []

    # Ensure positive integers
    if args.noise < 0:
        errors.append("Error: --noise must be a non-negative integer.")
    if args.chunksize <= 0:
        errors.append("Error: --chunksize must be a positive integer.")
    if args.seed_start < 0:
        errors.append("Error: --seed_start must be a non-negative integer.")

    # Check that input file exists
    if not os.path.exists(args.input):
        errors.append(f"Error: Input file '{args.input}' does not exist.")

    # Check output file overwrite rule
    if os.path.exists(args.output) and not args.force:
        errors.append(
            f"Error: Output file '{args.output}' already exists. Use --force to overwrite it.")

    if errors:
        raise ValueError("\n".join(errors))

    return args


def setup_logging(log_path, log_level=logging.INFO, overwrite_log=False):
    # Ensure the log directory exists
    file_path = Path(log_path)
    log_directory = file_path.parent
    log_directory.mkdir(parents=True, exist_ok=True)

    # Ensure the log file exists
    if overwrite_log:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")
    else:
        file_path.touch(exist_ok=True)

    # logger_name = f"logger_{file_path.parent.name}_{file_path.stem}"
    # logger = logging.getLogger(logger_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = True

    # Check if a handler for the same log file already exists
    existing_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve()
            if handler_path == file_path.resolve():
                existing_handler = handler
                break

    if not existing_handler:
        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8")
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] - %(levelname)-8s - %(message)s",
            datefmt="%Y-%b-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def add_poission_noise(inputfile, outputfile, lam, seed_start, chunksize):

    def _get_data_shape(filename):
        dset_path = "/entry_0000/measurement/data"
        with h5.File(filename, "r") as h5file:
            dset = h5file[dset_path]
            shape = dset.shape
            dtype = dset.dtype
        return shape, dtype

    def _load_data(filename, frame=None):
        with h5.File(filename, "r") as h5file:
            dset_path = "/entry_0000/measurement/data"
            if frame is None:
                data = h5file[dset_path][()]
                return data
            if isinstance(frame, int):
                data = h5file[dset_path][frame]
                return data
            if isinstance(frame, tuple):
                data = h5file[dset_path][frame[0]:frame[1]]
                return data
        return None

    logger = logging.getLogger(__name__)

    file_size_bytes = os.path.getsize(inputfile)
    file_size_mb = file_size_bytes / (1024 ** 2)
    file_size_gb = file_size_bytes / (1024 ** 3)
    logger.info(
        f"InputFile '{inputfile}' size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")

    with h5.File(inputfile, "r") as h:
        h5_data_path = "/entry_0000/measurement/data"

        input_data = h[h5_data_path]
        data_shape, dtype = _get_data_shape(inputfile)
        nframes = data_shape[0]
        frame_shape = data_shape[1:]

        with h5.File(outputfile, "w") as w:
            w.attrs["creator"] = "LIMA"
            w.attrs["default"] = "entry_0000"
            logger.info(f"number_of_frames: {nframes}")
            logger.info(f"frame_shape: ({frame_shape[1]},{frame_shape[0]})")
            logger.info(f"dtype: {dtype}")
            output_data = w.create_dataset(h5_data_path,
                                           shape=data_shape,
                                           dtype=dtype,
                                           chunks=(1,)+frame_shape,
                                           compression=hdf5plugin.Bitshuffle(),)

            rotmats = np.zeros(shape=(nframes,)+(3, 3), dtype=np.float32)
            for i in range(0, nframes//chunksize):
                logger.info(
                    f"Loading frames {i*chunksize+1:>5d}-{(i+1)*chunksize:>5d}")
                frames = _load_data(inputfile, frame=(
                    i*chunksize, (i+1)*chunksize))
                # check size of frames
                size_bytes = frames.nbytes
                size_mb = size_bytes / (1024 ** 2)
                size_gb = size_bytes / (1024 ** 3)
                logger.info(
                    f"Frames loaded,   memory usage: {size_mb:.2f} MB ({size_gb:.3f} GB)")

                np.random.seed(seed_start+i)
                noise_chunk = np.random.poisson(
                    lam=lam, size=(chunksize,)+frame_shape)

                size_bytes = noise_chunk.nbytes
                size_mb = size_bytes / (1024 ** 2)
                size_gb = size_bytes / (1024 ** 3)
                logger.info(
                    f"Noise generated, memory usage: {size_mb:.2f} MB ({size_gb:.3f} GB)")

                frames = frames + noise_chunk

                for idx, frame in enumerate(frames, start=i*chunksize):
                    output_data[idx] = (frame).astype(dtype)

        with h5.File(outputfile, "a") as f:
            dset_path = "/entry_0000/processing/rotmats"
            rotmats = h[dset_path][()]
            logger.info("Storing rotmats.")
            _ = f.create_dataset(
                dset_path,
                data=rotmats,
                dtype=np.float32,
                chunks=(1, 3, 3),
                compression=hdf5plugin.Bitshuffle(),
            )

            logger.info("Storing ishit labels.")
            _ = f.create_dataset(
                "/entry_0000/processing/peakfinder/isHit",
                data=np.full(nframes, fill_value=1, dtype=np.uint8),
                dtype=np.uint8,
                compression=hdf5plugin.Bitshuffle(),
            )
            logger.info("Done")

    file_size_bytes = os.path.getsize(outputfile)
    file_size_mb = file_size_bytes / (1024 ** 2)
    file_size_gb = file_size_bytes / (1024 ** 3)
    logger.info(
        f"OutputFile '{outputfile}' size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")


if __name__ == "__main__":
    main()
