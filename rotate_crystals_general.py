#!/usr/bin/env python3

import numpy as np
import sys
import copy
import argparse
import time
import json


def get_symmetry_operators(space_group, filename="pymatgen_spacegroup_symmetry_ops.json"):

    with open(filename, "r") as f:
        spacegroup_data = json.load(f)
    for number in list(spacegroup_data.keys()):
        if spacegroup_data[number]["symbol"]==space_group:
            space_group_number=number
            break

    print(f" Space group number: {space_group_number}")
    operators = []
    for op in list(spacegroup_data[space_group_number]["operations"].keys()):
        if spacegroup_data[space_group_number]["operations"][op] not in operators:
            operators.append(
                spacegroup_data[space_group_number]["operations"][op])

    operators = np.array(operators, dtype=np.int8)
    print(f" Number of unique symmetry operators: {len(operators)}")

    return operators



def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Standardise orientations and order of chunks in a stream file.",
        epilog="Tamas Foldes, ESRF - 2025-04-11")
    parser.add_argument("-i", "--streamfile", action="store", type=str, dest="streamfile", required=True,
                        help="Path to the stream file.")
    parser.add_argument("-s", "--spacegroup", action="store", type=str, dest="spacegroup", required=True,
                        help="Space group of the crystals.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        dest="output", help="Output file name.")
    parser.add_argument("-N", "--max-chunks", type=int, default=None,
                        dest="max_chunks", help="Number of chunks to process.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.streamfile.split(".")[0] + "_ordered.stream"
    return args


class Chunk():
    def __init__(self):
        self.lines = []
        self.image_serial = None
        self.starlines = [-1, -1, -1]
        self.N_crystals = 0

    def standardise_orientations(self, operators):
        self.rotate_crystals(operators)

    def __str__(self):
        return f"Chunk {self.image_serial} with {self.N_crystals} crystals"

    def __repr__(self):
        return f"Chunk {self.image_serial} with {self.N_crystals} crystals"

    def rotate_single_crystal(self, crystal, operators):
        reference = np.array([1.0, 1.0, 1.0])
        vector_line_indices = [-1, -1, -1]
        for i, line in enumerate(crystal):
            if line.startswith("astar ="):
                astar = np.array(line.split()[2:5], dtype=np.double)
                vector_line_indices[0] = i
            if line.startswith("bstar ="):
                bstar = np.array(line.split()[2:5], dtype=np.double)
                vector_line_indices[1] = i
            if line.startswith("cstar ="):
                cstar = np.array(line.split()[2:5], dtype=np.double)
                vector_line_indices[2] = i
            if line.startswith("   h    k    l"):
                hkl_start = i+1
            if line.startswith("End of reflections"):
                hkl_end = i
                break

        original_axes = np.array([astar, bstar, cstar], dtype=np.double)

        best_orientation = {"angle": 180.1}
        for operator in operators:
            transformed = self.linear_transform_no_multiplication(
                original_axes, operator)
            angle = self.angle_between_vectors(
                np.sum(transformed, axis=0), reference)
            angle = 90.0 - np.abs(90.0-angle)
            if angle < best_orientation["angle"]:
                best_orientation["angle"] = angle
                best_orientation["operator"] = operator
                best_orientation["axes"] = transformed

        newlines = copy.deepcopy(crystal)
        starlines = crystal[vector_line_indices[0]:vector_line_indices[2]+1]
        rotated_starlines = self.format_star_lines(
            starlines, best_orientation["axes"])
        for i, index in enumerate(vector_line_indices):
            newlines[index] = rotated_starlines[i]

        if hkl_start == hkl_end:
            return newlines

        reflections = crystal[hkl_start:hkl_end]
        transformed_reflections = self.transform_and_sort_lines_fast3(
            reflections, best_orientation["operator"])
        newlines[hkl_start:hkl_end] = transformed_reflections

        return newlines

    def rotate_crystals(self, operators):
        newchunk = []
        crystal = []
        iscrystal = False
        for line in self.lines:
            if line.startswith("--- Begin cryst"):
                iscrystal = True
            if line.startswith("--- End cryst"):
                iscrystal = False
                newlines = self.rotate_single_crystal(crystal, operators)
                newchunk = newchunk + newlines
                crystal = []
            if iscrystal == False:
                newchunk.append(line)
            else:
                crystal.append(line)
        self.lines = newchunk

    @staticmethod
    def angle_between_vectors(v1, v2):
        # Normalize both vectors to avoid numerical instability
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Guard against divide-by-zero or invalid values
        if norm_v1 == 0 or norm_v2 == 0:
            raise ValueError("Input vectors must be non-zero")

        # Clamp the cosine value to the valid range [-1, 1] to avoid NaNs
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)

        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    @staticmethod
    def multiply_without_operator(a, b):
        """
        Multiplies two integers without using the * operator.
        Supports only -1, 0, or 1 as values for 'a'.
        """
        if a == 0:
            return 0
        elif a == 1:
            return b
        elif a == -1:
            return -b
        else:
            raise ValueError("Operator matrix must contain only -1, 0, or 1.")

    @staticmethod
    def linear_transform_no_multiplication(matrix, operator):
        """
        Applies a linear transformation using the operator matrix (values -1, 0, or 1)
        to the given 3x3 matrix, without using multiplication.

        Parameters:
            matrix (np.ndarray): A 3x3 numpy array containing the original vectors.
            operator (np.ndarray): A 3x3 numpy array with values -1, 0, or 1.

        Returns:
            np.ndarray: The transformed 3x3 matrix.
        """
        result = np.zeros((3, 3), dtype=np.double)

        for i in range(3):
            for j in range(3):
                total = 0
                for k in range(3):
                    total += Chunk().multiply_without_operator(
                        operator[i][k], matrix[k][j])
                result[i][j] = total

        return result

    def format_star_lines(self, original_lines, new_values):
        """
        Overwrites numeric values in lines like 'astar = +0.1234567 +0.1234567 +0.1234567 nm^-1'
        with new values from a NumPy array, preserving formatting (including + signs).

        Parameters:
            original_lines (list of str): Original lines containing vector labels and values.
            new_values (np.ndarray): A 3x3 NumPy array with new numeric values.

        Returns:
            list of str: New lines with updated values, formatted to match the original.
        """
        formatted_lines = []

        for i, line in enumerate(original_lines):
            label = line.split('=')[0].strip()
            values = new_values[i]

            # Format each number to have a leading sign and 7 decimal places
            formatted_numbers = ['{:+0.7f}'.format(val) for val in values]
            new_line = f"{label} = {' '.join(formatted_numbers)} nm^-1\n"
            formatted_lines.append(new_line)

        return formatted_lines

    def transform_and_sort_lines(self, lines, operator):
        """
        - Converts each row (3 ints) to 3x3 diagonal matrix.
        - Applies linear transformation.
        - Extracts the diagonal.
        - Sorts and rebuilds lines with preserved formatting.
        """
        transformed_lines = []

        for line in lines:
            parts = line.strip().split()
            diag_values = list(map(int, parts[:3]))
            rest = line[14:]

            diag_matrix = np.diag(diag_values)
            # transformed_matrix = self.linear_transform_no_multiplication(
            #     diag_matrix, operator)
            transformed_matrix = operator @ diag_matrix
            new_diag = np.diag(transformed_matrix).astype(np.int16)

            transformed_lines.append((new_diag.tolist(), rest))

        # Sort by the transformed diagonal values
        transformed_lines.sort(key=lambda x: tuple(x[0]))

        # Format output
        result_lines = []
        for diag, rest in transformed_lines:
            line_str = f"{diag[0]:>4} {diag[1]:>4} {diag[2]:>4}{rest}"
            result_lines.append(line_str)

        return result_lines

    def transform_and_sort_lines_fast(self, lines, operator):
        """
        Fast version of transform_and_sort_lines:
        Uses only non-zero elements of the operator to compute the new diagonal,
        avoiding full matrix multiplication.
        """
        transformed_lines = []

        # Precompute non-zero indices for each row in operator
        non_zero_indices = [
            [j for j in range(3) if operator[i][j] != 0]
            for i in range(3)
        ]

        for line in lines:
            parts = line.strip().split()
            diag_values = list(map(int, parts[:3]))
            rest = line[14:]

            new_diag = []
            for i in range(3):
                val = 0
                for j in non_zero_indices[i]:
                    val += operator[i][j] * diag_values[j]
                new_diag.append(val)

            transformed_lines.append((new_diag, rest))

        # Sort by transformed diagonal
        transformed_lines.sort(key=lambda x: tuple(x[0]))

        # Format output
        result_lines = []
        for diag, rest in transformed_lines:
            line_str = f"{diag[0]:>4} {diag[1]:>4} {diag[2]:>4}{rest}"
            result_lines.append(line_str)

        return result_lines

    def transform_and_sort_lines_fast2(self, lines, operator):
        hkl = np.array([list(map(int, line.split()[:3]))
                       for line in lines]).astype(np.int8)
        hkl_reindexed = np.zeros_like(hkl).astype(np.int8)
        rest = [line[14:] for line in lines]
        for i in range(3):
            for j in range(3):
                if operator[i][j] != 0:
                    if operator[i][j] == 1:
                        hkl_reindexed[:, i] += hkl[:, j]
                    elif operator[i][j] == -1:
                        hkl_reindexed[:, i] -= hkl[:, j]
                    else:
                        raise ValueError(
                            "Operator matrix must contain only -1, 0, or 1.")
        order = np.lexsort(
            (hkl_reindexed[:, 2], hkl_reindexed[:, 1], hkl_reindexed[:, 0]))
        result_lines = []
        for i in order:
            result_lines.append(
                f"{hkl_reindexed[i, 0]:>4} {hkl_reindexed[i, 1]:>4} {hkl_reindexed[i, 2]:>4}{rest[i]}")
        return result_lines

    def transform_and_sort_lines_fast3(self, lines, operator):
        # Extract HKL integers from each line
        hkl = np.fromiter(
            (int(word) for line in lines for word in line.split()[:3]),
            dtype=int
        ).reshape(-1, 3)

        # Pre-allocate output
        hkl_reindexed = np.zeros_like(hkl)

        # Fast transform using only non-zero entries
        for i in range(3):
            op_row = operator[i]
            hkl_reindexed[:, i] = (
                (op_row[0] * hkl[:, 0]) +
                (op_row[1] * hkl[:, 1]) +
                (op_row[2] * hkl[:, 2])
            )

        # Prepare trailing text after first 3 numbers (avoid split overhead)
        rest = [line[14:] for line in lines]

        # Sort by transformed HKL values
        order = np.lexsort(
            (hkl_reindexed[:, 2], hkl_reindexed[:, 1], hkl_reindexed[:, 0]))

        # Format output efficiently
        result_lines = [
            f"{hkl_reindexed[i, 0]:>4} {hkl_reindexed[i, 1]:>4} {hkl_reindexed[i, 2]:>4}{rest[i]}"
            for i in order
        ]

        return result_lines

    def transform_and_sort_lines_fast4(self, lines, operator):
        lin_op = np.array([0, 0, 0], dtype=np.int8)
        ispositive = np.full(3, fill_value=True, dtype=bool)
        for i in range(3):
            for j in range(3):
                if operator[i][j] != 0:
                    lin_op[j] = i
                    ispositive[j] = operator[i][j] > 0
        hkl = np.array([list(map(int, line.split()[:3]))
                       for line in lines]).astype(np.int8)
        for i in range(3):
            if not ispositive[i]:
                hkl[:, lin_op[i]] *= -1
        order = np.lexsort(
            (hkl[:, lin_op[2]], hkl[:, lin_op[1]], hkl[:, lin_op[0]]))
        result_lines = []
        for i in order:
            result_lines.append(
                f"{hkl[i, lin_op[0]]:>4} {hkl[i, lin_op[1]]:>4} {hkl[i, lin_op[2]]:>4}{lines[order[i]][14:]}")
        return result_lines


def get_header(filename):
    with open(filename, 'r', buffering=1024 * 1024) as inpfile:
        header = []
        for line in inpfile:
            if line.startswith("----- Begin chunk"):
                break
            header.append(line)
    return header


def get_chunks(filename, N=None):
    with open(filename, 'r', buffering=1024 * 1024) as inpfile:
        chunks = {}
        temp = []
        N_crystals = 0
        ischunk = False
        for line in inpfile:
            if line.startswith("----- Begin chunk"):
                ischunk = True
            if ischunk:
                temp.append(line)
                if line.startswith("--- Begin cryst"):
                    N_crystals += 1
                if line.startswith("Image serial number:"):
                    image_serial = int(line.split()[-1])
            if line.startswith("----- End chunk"):
                newchunk = Chunk()
                newchunk.lines = temp
                newchunk.image_serial = image_serial
                newchunk.N_crystals = N_crystals
                chunks[image_serial] = newchunk
                temp = []
                if N is not None and len(list(chunks.keys())) >= N:
                    break
    return chunks


def write_stream(filename, header, chunks, ordered_keys) -> None:
    print(f" Writing stream {filename}")
    with open(filename, "w") as ofile:
        ofile.writelines(header)
        for i, key in enumerate(ordered_keys):
            print(f"{i/len(ordered_keys)*100:>4.1f}%", end="\r")
            chunklines = chunks[key].lines
            ofile.writelines(chunklines)
    print(f" Done writing output stream                            ")


def main():
    args = parse_args()

    operators = get_symmetry_operators(space_group=args.spacegroup)

    header = get_header(args.streamfile)
    chunks = get_chunks(args.streamfile, N=args.max_chunks)
    number_of_chunks = len(sorted(list(chunks.keys())))
    print(f" Number of collected chunks: {number_of_chunks}")

    print(" Transforming chunks now...   0.0%", end="\r")
    start = time.time()
    for k, key in enumerate(sorted(list(chunks.keys()))):
        chunks[key].standardise_orientations(operators)
        print(
            f" Transforming chunks now... {k/number_of_chunks*100:5>.1f}%      ", end="\r")
    print(" Done transforming chunks                        ")
    all_image_serials = sorted(list(chunks.keys()))

    write_stream(filename=args.output,
                 header=header,
                 chunks=chunks,
                 ordered_keys=all_image_serials)

    end = time.time()
    print(f" Total processing time: {end-start:.3f} seconds")


if __name__ == "__main__":
    main()




