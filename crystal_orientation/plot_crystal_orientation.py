import logging
import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import cKDTree


log_filename = "plot_crystal_orientations.log"
log_level = logging.INFO


def main():
    logger = setup_logging(
        log_path=log_filename,
        log_level=log_level,
        overwrite_log=True,
    )

    args = parse_arguments()

    logger.info(f"Analyzing stream file:\n {args.stream_file}")
    if args.maxpoints is None:
        logger.info("Plotting all orientations")
    else:
        logger.info(f"Plotting {args.maxpoints} number of points")

    rec_latt_vectors = _parse_star_lines_to_array(args.stream_file)
    logger.info(f"Number of crystals found: {np.shape(rec_latt_vectors)[0]}")

    euler_angles_list = _convert_Rstar_to_euler_angles(rec_latt_vectors)
    logger.info("Vectors converted to Euler angles")

    mpl.use("Agg")
    _combined_plot(euler_angles_list, args.png, NmaxPoints=args.maxpoints)
    logger.info(f"Plot saved at {args.png}")


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


def parse_arguments():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        prog="plot_crystal_orientation.py",
        description=(
            "Process a CrystFEL .stream file and generate a single PNG figure "
            "containing a Lambert projections of crystal orientations and "
            "Euler angle histograms. It uses Bunge XZX formalism."
            "\nTamas FOLDES - ESRF - 18/12/2025"
        ),
        epilog=(
            "Examples:\n"
            "  plot_crystal_orientation.py input.stream\n"
            "  plot_crystal_orientation.py input.stream -o orientations.png"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "stream_file",
        type=Path,
        help="Path to the CrystFEL .stream file"
    )

    parser.add_argument(
        "-p", "--png",
        type=Path,
        default=Path("crystal_orientation.png"),
        help="Output PNG file path (default: crystal_orientation.png)"
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite output PNG if it already exists (default)"
    )

    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite output PNG if it already exists"
    )

    parser.add_argument(
        "--maxpoints",
        type=int,
        default=None,
        help="Maximum number of points to plot from the stream file "
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # ---- Validation (raise regular exceptions) ----

    if not args.stream_file.is_file():
        logger.error(f"Stream file not found: {args.stream_file}")
        raise FileNotFoundError(
            f"Stream file not found: {args.stream_file}"
        )

    if args.png.suffix != ".png":
        logger.error(f"Invalid png filename: {args.png}")
        raise ValueError(
            f"Invalid png file extension: {args.png} "
            "(expected .png)"
        )

    if not args.png.parent.exists():
        logger.error(f"Invalid png path: {args.png}")
        raise FileNotFoundError(
            f"Output directory does not exist: {args.png.parent}"
        )

    if args.png.exists() and not args.overwrite:
        logger.error(f"NO overwrite and file already exists: {args.png}")
        raise FileExistsError(
            f"Output file already exists and overwrite is disabled: {args.png}"
        )

    return args


def _parse_star_lines_to_array(filepath, encoding="utf-8"):
    """
    Extracts 'astar', 'bstar', and 'cstar' lines from a binary text file,
    parses the 3x3 matrix of floats per group, and returns an Nx3x3 numpy array.
    Args:
        filepath (str): Path to the input file.
    Returns:
        np.ndarray: A NumPy array of shape (N, 3, 3).
    """
    # Regex to match 'astar =' in binary and extract floats
    pattern = re.compile(rb"^astar = ")

    matrices = []

    with open(filepath, "rb") as f:
        lines = iter(f)
        for line in lines:
            if pattern.match(line):
                try:
                    a_line = line
                    b_line = next(lines)
                    c_line = next(lines)
                except StopIteration:
                    # Incomplete list of reciprocal vectors e.g at end of file
                    break

                # Extract floats from all three lines
                a_vals = [val.decode(encoding) for val in a_line.split()[2:5]]
                b_vals = [val.decode(encoding) for val in b_line.split()[2:5]]
                c_vals = [val.decode(encoding) for val in c_line.split()[2:5]]
                matrix = [a_vals, b_vals, c_vals]
                matrices.append(matrix)

    matrices = np.array(matrices, dtype=np.double)  # Shape: (N, 3, 3)
    return matrices


def _convert_Rstar_to_euler_angles(rec_latt_vectors):
    """
    Given reciprocal lattice vectors a*, b*, c* (each a 3-element numpy array),
    generates the orientation matrix U (real-space normalized
    lattice vectors as columns). It finally return Euler angles (phi1, Phi, phi2)
    in degrees using Bunge ZXZ convention.
    """
    euler_angles_list = []
    for vectors in rec_latt_vectors:
        a_star = vectors[0]
        b_star = vectors[1]
        c_star = vectors[2]
        # Stack into reciprocal lattice matrix
        R_star = np.column_stack((a_star, b_star, c_star))

        # Compute real-space lattice matrix: R = (R*)^-T
        R = np.linalg.inv(R_star.T)

        # Normalize columns to get orientation only
        an = R[:, 0] / np.linalg.norm(R[:, 0])
        bn = R[:, 1] / np.linalg.norm(R[:, 1])
        cn = R[:, 2] / np.linalg.norm(R[:, 2])

        # Construct orientation matrix
        U = np.column_stack((an, bn, cn))

        # Extract elements
        u13, u23, u33 = U[0, 2], U[1, 2], U[2, 2]
        u31, u32 = U[2, 0], U[2, 1]

        # Compute Euler angles
        Phi = np.arccos(u33)

        if abs(u33) < 0.999999:  # General case
            phi1 = np.arctan2(u31, -u32)
            phi2 = np.arctan2(u13, u23)
        else:  # Gimbal lock (Phi = 0 or 180°)
            phi1 = np.arctan2(-U[1, 0], U[0, 0])
            phi2 = 0.0

        # Convert to degrees and wrap to [0, 360)
        phi1_deg = np.degrees(phi1) % 360
        Phi_deg = np.degrees(Phi) % 360
        phi2_deg = np.degrees(phi2) % 360

        # Append to the list
        euler_angles_list.append([phi1_deg, Phi_deg, phi2_deg])
    euler_angles_list = np.array(euler_angles_list)  # Shape: (N, 3)
    return euler_angles_list


def _euler_to_rotation_matrix(phi1, Phi, phi2):
    """Convert Bunge Euler angles (in degrees) to a rotation matrix."""
    phi1 = np.deg2rad(phi1)
    Phi = np.deg2rad(Phi)
    phi2 = np.deg2rad(phi2)

    Rz1 = np.array(
        [
            [np.cos(phi1), -np.sin(phi1), 0],
            [np.sin(phi1), np.cos(phi1), 0],
            [0, 0, 1],
        ]
    )

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(Phi), -np.sin(Phi)],
         [0, np.sin(Phi), np.cos(Phi)]]
    )

    Rz2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0],
            [np.sin(phi2), np.cos(phi2), 0],
            [0, 0, 1],
        ]
    )

    return Rz1 @ Rx @ Rz2


def _combined_plot(euler_angles_list, png_filename, NmaxPoints=None):
    """
    Visualizes Euler angle histograms and crystal orientation
    projections using Lambert projection.

    Parameters:
        euler_angles_list (np.ndarray): Array of shape (N, 3) containing
                                        Euler angles (phi1, Phi, phi2) in degrees.
        png_filename (str): Path of the generated png file.
    """

    # --- Helper Functions ---

    # def _normalize(vector):
    #     """Normalize a 3D vector."""
    #     return vector / np.linalg.norm(vector)

    def _rotation_matrix_from_vectors(source_vec, target_vec):
        """
        Returns the rotation matrix that aligns source_vec to target_vec.
        """
        a = source_vec / np.linalg.norm(source_vec)
        b = target_vec / np.linalg.norm(target_vec)
        # a, b = normalize(source_vec), normalize(target_vec)
        cross_prod = np.cross(a, b)
        dot_prod = np.dot(a, b)

        if np.allclose(cross_prod, 0):  # Vectors are parallel or anti-parallel
            return np.eye(3) if dot_prod > 0 else -np.eye(3)

        skew_sym_matrix = np.array(
            [
                [0, -cross_prod[2], cross_prod[1]],
                [cross_prod[2], 0, -cross_prod[0]],
                [-cross_prod[1], cross_prod[0], 0],
            ]
        )
        sin_angle = np.linalg.norm(cross_prod)
        return (
            np.eye(3)
            + skew_sym_matrix
            + skew_sym_matrix @ skew_sym_matrix *
            ((1 - dot_prod) / sin_angle**2)
        )

    def _count_neighbors(points: np.ndarray, d: float = 0.2) -> np.ndarray:
        """Count the neightbours around the points.
        This is used to assess relative density and to visualize it
        with a color map.
        """
        tree = cKDTree(points)
        counts = tree.query_ball_tree(tree, r=d)
        # counts[i] is a list of indices within
        # distance d from point i, including i itself
        # Subtract 1 to exclude the point itself
        neighbor_counts = np.array([len(c) - 1 for c in counts])
        return neighbor_counts

    def _lambert_projection(points):
        """Apply Lambert azimuthal equal-area projection to 3D points."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        denom = 1 + z
        valid = denom > 1e-8
        factor = np.zeros_like(denom)
        factor[valid] = np.sqrt(2 / denom[valid])
        return x * factor, y * factor

    # --- Visualization Settings ---

    angle_labels = [r"$\phi_1$", r"$\Phi$", r"$\phi_2$"]
    angle_limits = [[0, 360], [0, 180], [0, 360]]
    histogram_colors = [
        (19 / 255, 37 / 255, 119 / 255, 1),
        (183 / 255, 185 / 255, 186 / 255, 1),
        (237 / 255, 119 / 255, 3 / 255, 1),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 5.5),
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": [0.75, 4, 4, 4]},
        dpi=150,
    )

    # --- Euler Angle Histograms ---

    for col in range(1, 4):
        idx = col - 1
        bins = np.linspace(*angle_limits[idx], 61)
        angles = euler_angles_list[:, idx]
        hist_values, bin_edges = np.histogram(angles, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        axes[0, col].bar(
            bin_centers,
            hist_values,
            width=np.diff(bins)[0],
            color=histogram_colors[idx],
            label=angle_labels[idx],
        )
        axes[0, col].set_xticks(
            np.arange(0, angle_limits[idx][1] + 1, angle_limits[idx][1] // 6)
        )
        axes[0, col].set_xlim(*angle_limits[idx])
        axes[0, col].set_yticks([])
        axes[0, col].legend()

    # --- Compute Crystal Directions from Euler Angles ---

    np.random.seed(42)
    if NmaxPoints is not None:
        if NmaxPoints < len(euler_angles_list):
            indices = np.random.choice(euler_angles_list.shape[0], size=NmaxPoints, replace=False)
            euler_angles_list = euler_angles_list[indices]
            
    directions = np.eye(3)
    crystal_directions = np.array(
        [
            _euler_to_rotation_matrix(*angles) @ np.array([0, 0, 1])
            for angles in euler_angles_list
        ]
    )

    # --- Projection Grid Setup ---

    phi_angles = np.linspace(0, 2 * np.pi, 200)
    theta_angles = np.linspace(-np.pi / 2, np.pi / 2, 200)
    latitudes = np.deg2rad(np.arange(-90, 91, 30))
    longitudes = np.deg2rad(np.arange(0, 360, 30))

    # --- Color Coding Based on Neighbor Count ---

    neighbor_counts = _count_neighbors(crystal_directions, d=0.2)
    max_color_bins = 8
    color_values = neighbor_counts * \
        max_color_bins / (4 * np.pi) / max_color_bins
    color_max = np.quantile(neighbor_counts / (4 * np.pi), 0.75)
    color_min = 0

    # --- Plot Lambert Projections ---

    for i, projection_axis in enumerate(directions, start=1):
        rot_matrix = _rotation_matrix_from_vectors(projection_axis, [0, 0, 1])
        rotated_dirs = crystal_directions @ rot_matrix.T
        proj_x, proj_y = _lambert_projection(rotated_dirs)

        scatter = axes[1, i].scatter(
            proj_x,
            proj_y,
            s=2,
            alpha=0.75,
            c=color_values,
            cmap="rainbow",
            vmin=color_min,
            vmax=color_max,
            lw=0.0,
        )

        axes[1, i].text(
            -2.15, 2.15, f"Lambert prj.\n{projection_axis}", ha="left", va="top"
        )

        # Draw latitude lines
        for lat in latitudes:
            lat_circle = (
                np.stack(
                    (
                        np.cos(phi_angles) * np.cos(lat),
                        np.sin(phi_angles) * np.cos(lat),
                        np.full_like(phi_angles, np.sin(lat)),
                    ),
                    axis=-1,
                )
                @ rot_matrix.T
            )
            gx, gy = _lambert_projection(lat_circle)
            axes[1, i].plot(
                gx, gy, color="k", lw=0.5, alpha=0.6, ls=(0, (3.5, 2.0))
            )

        # Draw longitude lines
        for lon in longitudes:
            lon_line = (
                np.stack(
                    (
                        np.cos(theta_angles) * np.cos(lon),
                        np.cos(theta_angles) * np.sin(lon),
                        np.sin(theta_angles),
                    ),
                    axis=-1,
                )
                @ rot_matrix.T
            )
            gx, gy = _lambert_projection(lon_line)
            axes[1, i].plot(
                gx, gy, color="k", lw=0.5, alpha=0.6, ls=(0, (3.5, 2.0))
            )

        # Add boundary circle
        axes[1, i].add_patch(
            plt.Circle(
                (0, 0), 2, color="black", fill=False, lw=1.0, ls=(0, (3.5, 2.0))
            )
        )
        axes[1, i].set_aspect("equal", "box")

    # --- Final Plot Touches ---

    plt.suptitle(
        "Crystal Orientation Probability Distributions", y=0.94, fontsize=10
    )

    # Colorbar inset
    colorbar_ax = inset_axes(
        axes[1, 0],
        width="50%",
        height="90%",
        loc="lower left",
        bbox_to_anchor=(-0.60, 0.05, 0.5, 1),
        bbox_transform=axes[1, 0].transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(scatter, cax=colorbar_ax, extend="max")
    cbar.set_label(r"Relative angular density, $\rho/\rho_{av}$", fontsize=10)

    # Remove unused axes
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Explanatory text
    annotation_text = (
        "\n\nEuler angle\n"
        r"histograms $\to$"
        "\n\n\n"
        r"$\vec{c}$"
        " unit vector\n"
        "orientations\n"
        r"$\Rdsh$"
    )
    axes[0, 0].text(
        0.5, 0.5, annotation_text, ha="center", va="center", fontsize=10
    )

    plt.subplots_adjust(top=0.88)

    # plt.tight_layout()
    plt.savefig(png_filename, dpi=500, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
