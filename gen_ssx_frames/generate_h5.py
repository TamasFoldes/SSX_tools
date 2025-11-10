from cctbx import xray
from iotbx import pdb
from simtbx.diffBragg import utils
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from simtbx.nanoBragg.sim_data import SimData
from simtbx import nanoBragg
from scitbx.matrix import sqr
from dxtbx.model.beam import BeamFactory
from dxtbx.model import Crystal
from scipy.spatial.transform import Rotation
import sys
import os
import argparse
import multiprocessing as mp
import logging
import copy
from pathlib import Path
import numpy as np
import h5py as h5
import hdf5plugin
hdf5plugin.register()


def main():
    args = parse_args()

    # global logger
    logger = setup_logging(
        log_path=args.logfile,
        log_level=logging.INFO,
        overwrite_log=True,
    )

    crystal = SimFrame()

    logger.info(
        f"Generating {args.nframes} images as chunks of {args.chunksize}")
    logger.info(f"Creating file: {args.h5_file}")
    logger.info(f"First random seed: {args.seed_start}")

    wavelengths, wavelength_weights = crystal._get_gaussian_weights(
        fwhm=0.01,
        center=1.0725,
        N=7,
    )

    update_params = {
        "pdb_file": "MYO-spars_refine_064_full.pdb",
        "Ncells_abc": (100, 100, 25),
        "wavelengths": wavelengths,
        "wavelength_weights": wavelength_weights,
        "beam_size_mm": 0.0005,
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
        "-l", "--logfile",
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
        errors.append(
            f"Output file '{args.h5_file}' already exists. Use --force to overwrite.")

    if errors:
        # for e in errors:
        #     logger.error(e)
        raise ValueError(
            "Invalid command-line arguments:\n" + "\n".join(errors))

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
            logger.warning(
                f"Value {old_val} at index {i} clipped to {new_val}")

    # Apply clipping
    array = np.clip(array, min_val, max_val)

    return array.astype(dtype)


class SimFrame:
    def __init__(self):
        self.pdb_file = None
        self.wavelengths = [1.0725]
        self.wavelength_weights = None
        self.Ncells_abc = (3, 3, 3)
        self.pixelsize_mm = 0.075
        self.detector_distance_mm = 100
        self.pixelsize_mm = 0.075
        self.image_shape = (2164, 2068)
        self.rotmat = np.eye(3).astype(np.float32)
        self.img = None
        self.polarization_fraction = 0.99
        self.beam_size_mm = 2.83/1000
        self.flux = 1.0e11  # photons per pulse
        self.anomalous = True
        self.use_cuda = False

    def _get_gaussian_weights(self, fwhm, N, center):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        x_points = np.linspace(-2.5*sigma, 2.5*sigma, N)
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

    def _random_rotmat(self, seed=42):
        self.rotmat = Rotation.random(random_state=seed).as_matrix()

    # def _gen_structure_factors_old(self):
    #     F = utils.get_complex_fcalc_from_pdb(self.pdb_file).as_amplitude_array()
    #     return F

    def _gen_structure_factors_anom(self, wavelength, anomalous_flag=True, d_min=1.0):
        pdb_inp = pdb.input(self.pdb_file)
        xrs = pdb_inp.xray_structure_simple()

        if anomalous_flag:
            # Use wavelength-dependent f′/f″
            xrs.set_inelastic_form_factors(wavelength, "sasaki")
            fcalc = xrs.structure_factors(
                anomalous_flag=True, d_min=d_min).f_calc()
        else:
            # Normal scattering (no dispersion corrections)
            fcalc = xrs.structure_factors(
                anomalous_flag=False, d_min=d_min).f_calc()

        # <-- convert to amplitudes before passing to nanoBragg
        F = fcalc.as_amplitude_array()
        return F

    def _gen_single_wavelength_image(self, wavelength=1.0725, anomalous=True):

        if self.pdb_file is None:
            raise ValueError("No pdb file defined.")

        F = self._gen_structure_factors_anom(
            wavelength,
            anomalous_flag=anomalous,
        )

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

        if anomalous:
            # Enable anomalous scattering
            S.D.anomalous_flag = True
            S.D.wavelength_A = wavelength
        else:
            # Disable anomalous scattering
            S.D.anomalous_flag = False

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

    def _gen_multi_wavelength_image(self, weights=None, anomalous=True):
        if weights is None:
            weights = np.full(shape=len(self.wavelengths), fill_value=1.0)
        for cnt, (wavelength, weight) in enumerate(zip(self.wavelengths, weights)):
            tmp = self._gen_single_wavelength_image(
                wavelength=wavelength,
                anomalous=anomalous,
            )
            if cnt == 0:
                img = tmp * weight
            else:
                img += tmp * weight
        self.img = img


def _generate_frame(args):
    """Worker function to generate one frame and rotmat."""
    crystal_template, seed = args
    p = copy.deepcopy(crystal_template)
    p._random_rotmat(seed=seed)
    p._gen_multi_wavelength_image(
        weights=p.wavelength_weights,
        anomalous=p.anomalous
    )
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
    tmp_dir=None,
):
    """Generate synthetic diffraction frames, cache them in chunks,
    and merge into one HDF5 file."""

    logger = logging.getLogger(__name__)

    # --- Helper functions ---

    def _get_tmp_dir(h5_file, tmp_dir):
        """
        Determine and create the temporary chunk directory.

        If tmp_dir is not specified, create a folder in the same directory
        as the h5_file, named '<h5_file_stem>_tmp'.
        """
        h5_path = Path(h5_file)
        if tmp_dir is None:
            tmp_dir = h5_path.parent / f"{h5_path.stem}_tmp"
        else:
            tmp_dir = Path(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def _update_template_params(p, update_params):
        """Apply optional parameter updates to the crystal template."""
        if update_params:
            lines = "\n".join(f"  {k}: {v}" for k, v in update_params.items())
            logger.info("Updating parameters:\n" + lines)
            p._update_params(update_params)
        return p

    def _generate_chunk_file(chunk_idx, start_seed):
        """Generate one chunk of frames and save as temporary HDF5 file."""
        chunk_file = Path(tmp_dir) / f"chunk_{chunk_idx:04d}.h5"
        if chunk_file.exists():

            try:
                nframes_in_h5 = _get_number_of_frames_in_h5_file(chunk_file)
                nrotmats_in_h5 = _get_number_of_rotmats_in_h5_file(chunk_file)
            except:
                nframes_in_h5 = 0
                nrotmats_in_h5 = 0

            if nframes_in_h5 == chunksize and nrotmats_in_h5 == chunksize:
                logger.info(
                    f"Chunk {chunk_file} already exists, skipping generation.")
                return chunk_file

        logger.info(
            f"Generating chunk {chunk_idx}:"
            f" frames {chunk_idx*chunksize + 1:>4d}"
            f" – {(chunk_idx+1)*chunksize:>4d}"
        )
        frames, rotmats = gen_chunks(
            crystal_template=p,
            chunksize=chunksize,
            seed_start=start_seed,
            dtype=dtype,
            nthreads=nthreads,
        )

        size_mb = frames.nbytes / (1024 ** 2)
        logger.info(f"Chunk {chunk_idx} ready, memory usage: {size_mb:.2f} MB")

        with h5.File(chunk_file, "w") as cf:
            cf.attrs["creator"] = "LIMA"
            cf.attrs["default"] = "entry_0000"
            cf.create_dataset(
                "/entry_0000/measurement/data",
                data=frames.astype(dtype),
                dtype=dtype,
                compression=hdf5plugin.Bitshuffle()
            )
            cf.create_dataset(
                "/entry_0000/processing/rotmats",
                data=rotmats.astype(np.float32),
                dtype=np.float32,
                compression=hdf5plugin.Bitshuffle()
            )
        return chunk_file

    def _combine_chunks(output_file, chunk_files):
        """Combine all chunk files into the final HDF5 output file."""
        logger.info(f"Combining {len(chunk_files)} chunks into {output_file}")
        with h5.File(output_file, "w") as w:
            w.attrs["creator"] = "LIMA"
            w.attrs["default"] = "entry_0000"

            # Determine shape from the first chunk
            with h5.File(chunk_files[0], "r") as cf:
                n_chunk, * \
                    frame_shape = cf["/entry_0000/measurement/data"].shape
                dtype_local = cf["/entry_0000/measurement/data"].dtype

            total_frames = len(chunk_files) * n_chunk
            output_data = w.create_dataset(
                "/entry_0000/measurement/data",
                shape=(total_frames,) + tuple(frame_shape),
                dtype=dtype_local,
                chunks=(1,) + tuple(frame_shape),
                compression=hdf5plugin.Bitshuffle(),
            )
            rotmats_data = np.zeros((total_frames, 3, 3), dtype=np.float32)

            # Fill combined dataset
            idx = 0
            for chunk_file in chunk_files:
                with h5.File(chunk_file, "r") as cf:
                    frames = cf["/entry_0000/measurement/data"][()]
                    rotmats = cf["/entry_0000/processing/rotmats"][()]
                    n_chunk = frames.shape[0]
                    output_data[idx:idx+n_chunk] = frames
                    rotmats_data[idx:idx+n_chunk] = rotmats
                    idx += n_chunk

            # Add rotmats and isHit labels
            w.create_dataset(
                "/entry_0000/processing/rotmats",
                data=rotmats_data,
                dtype=np.float32,
                chunks=(1, 3, 3),
                compression=hdf5plugin.Bitshuffle(),
            )
            w.create_dataset(
                "/entry_0000/processing/peakfinder/isHit",
                data=np.full(total_frames, 1, dtype=np.uint8),
                dtype=np.uint8,
                compression=hdf5plugin.Bitshuffle(),
            )

        logger.info("Merging complete.")
        return output_file

    def _get_number_of_frames_in_h5_file(filename):
        dset_path = "/entry_0000/measurement/data"
        with h5.File(filename, "r") as h5file:
            dset = h5file[dset_path]
            shape = dset.shape
        return shape[0]

    def _get_number_of_rotmats_in_h5_file(filename):
        dset_path = "/entry_0000/processing/rotmats"
        with h5.File(filename, "r") as h5file:
            dset = h5file[dset_path]
            shape = dset.shape
        return shape[0]

    def _cleanup_tmp_files(h5_file, tmp_dir):
        """
        Remove all temporary files in tmp_dir and then remove the directory itself.
        Uses pathlib only (no shutil).
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaning up temporary directory: {tmp_dir}")

        h5_file_path = Path(h5_file)
        if not h5_file_path.is_file():
            logger.warning(f"No h5 file found: {h5_file}")
            return

        try:
            nframes_in_h5 = _get_number_of_frames_in_h5_file(h5_file)
            nrotmats_in_h5 = _get_number_of_rotmats_in_h5_file(h5_file)
        except:
            logger.warning(f"HDF5 file {h5_file} seems corrupted.")
            return
        if nframes_in_h5 != nframes or nrotmats_in_h5 != nframes:
            logger.warning(f"HDF5 file {h5_file} is incomplete.")
            return

        tmp_path = Path(tmp_dir)
        if not tmp_path.exists():
            logger.info(
                "Temporary directory does not exist, skipping cleanup.")
            return

        try:
            for child in tmp_path.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                        logger.debug(f"Deleted file: {child}")
                    elif child.is_dir():
                        # Remove empty subdirectories (shouldn't exist normally)
                        child.rmdir()
                        logger.debug(f"Removed subdirectory: {child}")
                except Exception as e:
                    logger.warning(f"Could not delete {child}: {e}")

            tmp_path.rmdir()
            logger.info(f"Removed temporary directory: {tmp_path}")
        except Exception as e:
            logger.warning(
                f"Could not remove temporary directory {tmp_dir}: {e}")

        logger.info("Cleanup complete.")

    # --- Main procedure ---
    tmp_dir = _get_tmp_dir(h5_file, tmp_dir)

    p = copy.deepcopy(crystal_template)
    p = _update_template_params(p, update_params)

    n_chunks = nframes // chunksize
    chunk_files = []

    # Generate (or reuse) chunk files
    for i in range(n_chunks):
        chunk_file = _generate_chunk_file(
            i, start_seed=i*chunksize + seed_start)
        chunk_files.append(chunk_file)

    # Combine chunks
    final_file = _combine_chunks(h5_file, chunk_files)

    # Log file size
    size_mb = os.path.getsize(final_file) / (1024 ** 2)
    size_gb = size_mb / 1024
    logger.info(
        f"Final file '{final_file}' size: {size_mb:.2f} MB ({size_gb:.3f} GB)")

    # Cleanup temporary files
    _cleanup_tmp_files(h5_file, tmp_dir)


if __name__ == "__main__":
    main()
