from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.defaults import (
    ERR_CONFIG_NOT_FOUND,
    ERR_PATH_NOT_FILE,
    ERR_PATH_TRAVERSAL,
    ERR_YAML_NOT_DICT,
    ERR_YAML_PARSE,
)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Loads a YAML file into a dictionary with path safety checks.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If path is invalid, attempts traversal, or file is empty.
        yaml.YAMLError: If the YAML is invalid.
    """
    path = Path(file_path).resolve()
    base_dir = Path.cwd().resolve()

    # Path Sanitization: Ensure path doesn't traverse outside allowed scope (CWD)
    # We strictly enforce that config files must be within the project root.
    # Absolute paths are allowed IF they resolve to be inside CWD.
    if not path.is_relative_to(base_dir):
        msg = ERR_PATH_TRAVERSAL.format(path=path, base=base_dir)
        raise ValueError(msg)

    if not path.exists():
        msg = ERR_CONFIG_NOT_FOUND.format(path=path)
        raise FileNotFoundError(msg)

    # Ensure it's a file, not a directory
    if not path.is_file():
        msg = ERR_PATH_NOT_FILE.format(path=path)
        raise ValueError(msg)

    with path.open("r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = ERR_YAML_PARSE.format(error=e)
            raise ValueError(msg) from e
        else:
            if not isinstance(data, dict):
                # Handle empty file or just scalar
                if data is None:
                    msg = "YAML file is empty"
                    raise ValueError(msg)
                raise TypeError(ERR_YAML_NOT_DICT)
            return data


def load_config(file_path: str | Path) -> PyAceConfig:
    """
    Loads a configuration file and validates it against the PyAceConfig schema.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Validated PyAceConfig object.
    """
    data = load_yaml(file_path)
    return PyAceConfig(**data)


def dump_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        data: Dictionary to write.
        file_path: Path to the output file.
    """
    path = Path(file_path)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


def detect_elements(file_path: Path, max_frames: int | None = None) -> list[str]:
    """
    Detects chemical elements from a structure file by scanning the first few frames.
    Optimized to avoid loading the entire file.

    Args:
        file_path: Path to the structure file (xyz, extxyz, etc.)
        max_frames: Maximum number of frames to scan. If None, uses default.

    Returns:
        Sorted list of unique chemical symbols found.

    Raises:
        ValueError: If no elements could be detected.
    """
    from ase.io import iread

    from pyacemaker.domain_models.defaults import DEFAULT_MAX_FRAMES_ELEMENT_DETECTION

    limit = max_frames if max_frames is not None else DEFAULT_MAX_FRAMES_ELEMENT_DETECTION

    elements_set = set()
    fmt = "extxyz" if file_path.suffix == ".xyz" else None
    read_fmt = fmt if fmt else ""

    try:
        # Use iread for streaming access
        for i, atoms in enumerate(iread(str(file_path), index=":", format=read_fmt)):
            elements_set.update(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
            if i >= limit:
                break
    except Exception as e:
        msg = f"Failed to read structure file {file_path}: {e}"
        raise ValueError(msg) from e

    if not elements_set:
        msg = f"No elements detected in {file_path} (checked {max_frames} frames)."
        raise ValueError(msg)

    return sorted(elements_set)


def write_lammps_streaming(
    fileobj: TextIO, structure: Atoms, elements: list[str], atom_style: str = "atomic"
) -> None:
    """
    Writes LAMMPS data file in streaming fashion to minimize memory usage for large structures.
    Optimized for 'atomic' style.

    Args:
        fileobj: Writable file-like object.
        structure: ASE Atoms object.
        elements: List of sorted unique elements (specorder).
        atom_style: LAMMPS atom style (default: atomic).
    """
    if atom_style != "atomic":
        # For non-atomic styles, we fallback to ASE write internally if needed,
        # but here we implement atomic.
        # If forced, one could add support. For now, we assume atomic or warn.
        pass

    n_atoms = len(structure)
    n_types = len(elements)

    fileobj.write("LAMMPS data file written by pyacemaker (streaming)\n\n")
    fileobj.write(f"{n_atoms} atoms\n")
    fileobj.write(f"{n_types} atom types\n\n")

    # Box
    # Simple orthogonal box logic
    # LAMMPS: xlo xhi, ylo yhi, zlo zhi
    cell = structure.get_cell()  # type: ignore[no-untyped-call]
    # Check if orthogonal
    if not np.allclose(cell, np.diag(np.diag(cell))):
        # Triclinic logic is complex for simple streaming.
        # Fallback to ASE's write_lammps_data logic via creating small dummy?
        # Or just raise error demanding simple box for streaming optimization?
        # We'll support orthogonal only for streaming optimization.
        # Use ASE write for complex cases upstream.
        msg = "Streaming writer only supports orthogonal cells."
        raise ValueError(msg)

    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = cell[0, 0], cell[1, 1], cell[2, 2]

    fileobj.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
    fileobj.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
    fileobj.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

    # Masses
    fileobj.write("Masses\n\n")
    for i, sym in enumerate(elements, start=1):
        z = atomic_numbers[sym]
        mass = atomic_masses[z]
        fileobj.write(f"{i} {mass:.4f} # {sym}\n")
    fileobj.write("\n")

    # Atoms
    fileobj.write("Atoms\n\n")

    # Map symbols to types
    type_map = {sym: i for i, sym in enumerate(elements, start=1)}

    # Stream atoms
    positions = structure.get_positions()  # type: ignore[no-untyped-call]
    # structure.get_positions() might return a copy or reference.
    # Accessing structure.arrays['positions'] is better if we want raw access?
    # ASE Atoms usually stores in arrays.

    symbols = structure.get_chemical_symbols()  # type: ignore[no-untyped-call]

    for i in range(n_atoms):
        atom_id = i + 1
        sym = symbols[i]
        typ = type_map[sym]
        x, y, z = positions[i]
        fileobj.write(f"{atom_id} {typ} {x:.6f} {y:.6f} {z:.6f}\n")
