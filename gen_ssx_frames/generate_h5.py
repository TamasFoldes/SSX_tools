import numpy as np
import h5py as h5
import hdf5plugin
hdf5plugin.register()
from pathlib import Path
import copy
import logging
import multiprocessing as mp
import argparse
import os
import sys

from scipy.spatial.transform import Rotation

from dxtbx.model import Crystal
from dxtbx.model.beam import BeamFactory
from scitbx.matrix import sqr
from simtbx import nanoBragg
from simtbx.nanoBragg.sim_data import SimData
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.diffBragg import utils


def main():
    args=parse_args()

    # global logger
    logger = setup_logging(
        log_path=args.logfile,
        log_level=logging.INFO,
        overwrite_log=True,
    )

    crystal=SimFrame()

    logger.info(f"Generating {args.nframes} images as chunks of {args.chunksize}")
    logger.info(f"Creating file: {args.h5_file}")
    logger.info(f"First random seed: {args.seed_start}")

    wavelengths, wavelength_weights = crystal._get_gaussian_weights(
        fwhm=0.01,
        center=1.0725,
        N=7,
    )

    update_params={
        "pdb_file":"MYO-spars_refine_064_full.pdb",
        "Ncells_abc":(100,100,25),
        "wavelengths":wavelengths,
        "wavelength_weights":wavelength_weights,
        "beam_size_mm":0.0005,
    }

    gen_and_save_frames(
        h5_file=args.h5_file,
        crystal_template=crystal,
        nframes=args.nframes,
        chunksize=args.chunksize,
        update_params=update_params,
        seed_start=args.seed_start,
        nthreads=args.nthreads,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=(
            "=== Parallel SSX diffraction image creator ===\n"
            "This tool processes a pdb file and generates diffraction images\n"
            "with random crystal orientations. The results are saved to an HDF5 file.\n"
            "You can control threading, frame count, and chunking behavior.\n"
        ),
        epilog=(
            "Example usage:\n"
            f"  python {sys.argv[0]} -t 8 -f 1000 -c 100 -o results.h5\n\n"
            "Use --force to overwrite an existing output file.\n"
            "==========================================="
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-t", "--nthreads",
        type=int,
        default=16,
        help="Number of threads to use for processing. (default: %(default)s)"
    )

    parser.add_argument(
        "-f", "--nframes",
        type=int,
        default=400,
        help="Total number of frames to process. (default: %(default)s)"
    )

    parser.add_argument(
        "-c", "--chunksize",
        type=int,
        default=100,
        help="Number of frames to process per chunk. (default: %(default)s)"
    )

    parser.add_argument(
        "-s", "--seed_start",
        type=int,
        default=0,
        help="Starting random seed value. (default: %(default)s)"
    )

    parser.add_argument(
        "-o", "--h5_file",
        type=str,
        default="testdata.h5",
        help="Path to the output HDF5 file. (default: %(default)s)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output file. (default: False)"
    )

    parser.add_argument(
        "-l","--logfile",
        type=str,
        default="h5_generation.log",
        help="Name of the logfile. (default: %(default)s)"
    )

    args = parser.parse_args()

    # --- Validation ---
    errors = []

    if args.nthreads <= 0:
        errors.append("nthreads must be a positive integer.")
    if args.nframes <= 0:
        errors.append("nframes must be a positive integer.")
    if args.chunksize <= 0:
        errors.append("chunksize must be a positive integer.")
    if args.nframes % args.chunksize != 0:
        errors.append("nframes must be divisible by chunksize.")

    # Check file existence
    if os.path.exists(args.h5_file) and not args.force:
        errors.append(f"Output file '{args.h5_file}' already exists. Use --force to overwrite.")

    if errors:
        # for e in errors:
        #     logger.error(e)
        raise ValueError("Invalid command-line arguments:\n" + "\n".join(errors))

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
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] - %(levelname)-8s - %(message)s",
            datefmt="%Y-%b-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def safe_cast(array: np.ndarray, dtype: np.dtype, verbose: bool = False) -> np.ndarray:
        """
        Safely cast a NumPy array to a specified dtype.
        Clips values to the valid range of the dtype to avoid infs or overflows.
        Optionally prints out any values that were changed.

        Parameters
        ----------
        array : np.ndarray
            Input array to be converted.
        dtype : np.dtype or type
            Target data type (e.g., np.uint8, np.int16, np.float32).
        verbose : bool, optional
            If True, prints when values are clipped or replaced. Default is False.

        Returns
        -------
        np.ndarray
            Array safely cast to the specified dtype.
        """
        logger = logging.getLogger(__name__)
        dtype = np.dtype(dtype)

        # Keep original for comparison
        original = np.copy(array)

        # Replace NaN and infinities with large/small finite numbers
        array = np.nan_to_num(array, nan=0.0, posinf=np.inf, neginf=-np.inf)

        # Clip to dtype range if needed
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            min_val, max_val = info.min, info.max
        elif np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            min_val, max_val = info.min, info.max
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

        # Identify out-of-range elements
        mask_low = array < min_val
        mask_high = array > max_val
        mask_invalid = ~np.isfinite(original)

        if verbose and (mask_low.any() or mask_high.any() or mask_invalid.any()):
            indices = np.where(mask_low | mask_high | mask_invalid)
            for i in zip(*indices):
                old_val = original[i]
                new_val = (
                    min_val if mask_low[i]
                    else max_val if mask_high[i]
                    else 0.0 if mask_invalid[i]
                    else array[i]
                )
                # print(f"Value {old_val} at index {i} clipped to {new_val}")
                logger.warning(f"Value {old_val} at index {i} clipped to {new_val}")

        # Apply clipping
        array = np.clip(array, min_val, max_val)

        return array.astype(dtype)


class SimFrame:
    def __init__(self):
        self.pdb_file=None
        self.wavelengths=[1.0725]
        self.wavelength_weights=None
        self.Ncells_abc=(3,3,3)
        self.pixelsize_mm=0.075
        self.detector_distance_mm=100
        self.pixelsize_mm=0.075
        self.image_shape=(2164,2068)
        self.rotmat=np.eye(3).astype(np.float32)
        self.img=None
        self.polarization_fraction=0.99
        self.beam_size_mm=2.83/1000
        self.flux=2.0e11 # photons per pulse
        self.use_cuda=False


    def _get_gaussian_weights(self, fwhm, N, center):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        x_points=np.linspace(-2.5*sigma,2.5*sigma,N)
        y_points = np.exp(-0.5 * (x_points / sigma) ** 2)
        wavelengths = x_points+center
        wavelength_weights = y_points
        return wavelengths, wavelength_weights


    def _update_params(self, params: dict, *, allow_new: bool = False) -> None:
        """
        Internal method to update multiple parameters from a dictionary.

        Args:
            params (dict): A dictionary of attribute-value pairs to update.
            allow_new (bool): If True, new attributes will be created.
                              If False, only existing attributes are updated.
        """
        for key, value in params.items():
            if hasattr(self, key) or allow_new:
                setattr(self, key, value)


    def _random_rotmat(self,seed=42):
        self.rotmat=Rotation.random(random_state=seed).as_matrix()


    def _gen_single_wavelength_image(self, wavelength=1.0725):

        if self.pdb_file is None:
            raise ValueError("No pdb file defined.")

        F = utils.get_complex_fcalc_from_pdb(self.pdb_file).as_amplitude_array()

        # Detector and beam setup
        dxtbx_det = SimData.simple_detector(
            detector_distance_mm=self.detector_distance_mm,
            pixelsize_mm=self.pixelsize_mm,
            image_shape=self.image_shape,
        )
        dxtbx_beam = BeamFactory.simple(wavelength=wavelength)

        # Crystal setup
        uc = F.crystal_symmetry().unit_cell()
        B = np.reshape(uc.orthogonalization_matrix(), (3, 3))
        a, b, c = B.T
        sym = F.crystal_symmetry().space_group().type().lookup_symbol()
        dxtbx_cryst = Crystal(a, b, c, sym)

        cryst = NBcrystal()
        cryst.Ncells_abc = self.Ncells_abc
        cryst.dxtbx_crystal = dxtbx_cryst
        cryst.miller_array = F

        # Beam configuration
        beam = NBbeam()
        beam.polarization_fraction = self.polarization_fraction
        beam.size_mm = self.beam_size_mm
        beam.unit_s0 = dxtbx_beam.get_unit_s0()
        beam.spectrum = [(dxtbx_beam.get_wavelength(), self.flux)]

        # Simulation
        S = SimData()
        S.detector = dxtbx_det
        S.crystal = cryst
        S.beam = beam
        S.instantiate_nanoBragg()

        # Orientation and image generation
        S.D.raw_pixels *= 0
        dxtbx_cryst.set_U(tuple(self.rotmat.ravel()))
        S.D.Amatrix = sqr(dxtbx_cryst.get_A()).transpose()

        if self.use_cuda:
            S.D.add_nanoBragg_spots_cuda()
        else:
            S.D.add_nanoBragg_spots()

        # Transpose the matrix for storing
        result = S.D.raw_pixels.as_numpy_array().T

        return result


    def _gen_multi_wavelength_image(self, weights=None):
        if weights is None:
            weights = np.full(shape=len(self.wavelengths), fill_value=1.0)
        for cnt, (wavelength,weight) in enumerate(zip(self.wavelengths,weights)):
            tmp=self._gen_single_wavelength_image(wavelength=wavelength)
            if cnt==0: 
                img=tmp * weight
            else:
                img+=tmp * weight
        self.img=img


def _generate_frame(args):
    """Worker function to generate one frame and rotmat."""
    crystal_template, seed = args
    p = copy.deepcopy(crystal_template)
    p._random_rotmat(seed=seed)
    p._gen_multi_wavelength_image(weights=p.wavelength_weights)
    return p.img, p.rotmat


def gen_chunks(crystal_template, chunksize, seed_start=0, dtype=np.int32, nthreads=10):
    """Parallel version of gen_chunks."""
    seeds = [seed_start + i for i in range(chunksize)]
    args = [(crystal_template, seed) for seed in seeds]
    with mp.Pool(processes=nthreads) as pool:
        results = pool.map(_generate_frame, args)
    frames, rotmats = zip(*results)
    frames = safe_cast(
        array=frames,
        dtype=dtype,
        verbose=True,
    )
    rotmats = np.stack(rotmats).astype(np.float32)
    return frames, rotmats


def gen_and_save_frames(
        h5_file,
        crystal_template,
        nframes,
        chunksize,
        update_params=None,
        dtype=np.int32,
        seed_start=0,
        nthreads=10,
    ):

    logger = logging.getLogger(__name__)

    p=copy.deepcopy(crystal_template)
    if update_params:
        logger.info(f"Updating the following parameters:\n{update_params}")
        p._update_params(update_params)

    with h5.File(h5_file, "w") as w:
        w.attrs["creator"] = "LIMA"
        w.attrs["default"] = "entry_0000"
        logger.info(f"number_of_frames: {nframes}")
        logger.info(f"frame_shape: {p.image_shape}")
        logger.info(f"dtype: {dtype}")

        output_data = w.create_dataset(
            "/entry_0000/measurement/data",
            shape=(nframes,)+(p.image_shape),
            dtype=dtype,
            chunks=(1,)+(p.image_shape),
            compression=hdf5plugin.Bitshuffle(),
        )

        rotmats=np.zeros(shape=(nframes,)+(3,3),dtype=np.float32)
        for i in range(0,nframes//chunksize):
            logger.info(f"Generating frames {i*chunksize+1:>5d}-{(i+1)*chunksize:>5d}")
            frames,rotmats_chunk=gen_chunks(
                crystal_template=p,
                chunksize=chunksize,
                seed_start=i*chunksize+seed_start,
                dtype=dtype,
                nthreads=nthreads,
            )
            # check size of frames
            size_bytes = frames.nbytes
            size_mb = size_bytes / (1024 ** 2)
            size_gb = size_bytes / (1024 ** 3)
            logger.info(f"Frames ready, memory usage: {size_mb:.2f} MB ({size_gb:.3f} GB)")
            for idx, (frame,rotmat) in enumerate(zip(frames,rotmats_chunk),start=i*chunksize):
                output_data[idx] = (frame).astype(dtype)
                rotmats[idx] = rotmat
        
    with h5.File(h5_file, "a") as f:
        logger.info("Storing rotmats.")
        _ = f.create_dataset(
            "/entry_0000/processing/rotmats",
            data=rotmats,
            dtype=np.float32,
            chunks=(1,3,3),
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

    file_size_bytes = os.path.getsize(h5_file)
    file_size_mb = file_size_bytes / (1024 ** 2)
    file_size_gb = file_size_bytes / (1024 ** 3)
    logger.info(f"File '{h5_file}' size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")


if __name__=="__main__":
    main()
    