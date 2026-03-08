import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import iread

logger = logging.getLogger(__name__)

# Cache atomic masses to avoid repeated imports/lookups in inner loops
_ATOMIC_MASSES_CACHE: dict[str, float] = {}


def load_yaml(filepath: Path) -> dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Dictionary containing configuration.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with filepath.open("r") as f:
        return yaml.safe_load(f) or {}

# Alias for backward compatibility
load_config = load_yaml

def detect_elements(data_path: Path, max_frames: int = 10) -> list[str]:
    """
    Detects elements present in the dataset by reading frames.

    Args:
        data_path: Path to the dataset file (xyz, extxyz, etc).
        max_frames: Max number of frames to check (default: 10).

    Returns:
        List of chemical symbols (sorted alphabetically).
    """
    symbols: set[str] = set()
    try:
        # Optimization: Use iread to peek. Stop if we have 'enough' frames or symbols stabilize?
        # Difficult to know if symbols stabilize. Just read max_frames.
        gen = iread(str(data_path), index=f":{max_frames}")
        for atoms in gen:
            if isinstance(atoms, Atoms):
                new_syms = set(atoms.get_chemical_symbols()) # type: ignore[no-untyped-call]
                # If we found new symbols, update.
                if not new_syms.issubset(symbols):
                    symbols.update(new_syms)
    except Exception:
        logger.warning(f"Could not fully read {data_path} to detect elements. Elements detected so far: {symbols}")

    return sorted(symbols)


def dump_yaml(data: Any, filepath: Path) -> None:
    """
    Dumps data to a YAML file safely.

    Args:
        data: The data to dump (dict, list, etc).
        filepath: Path to the output file.
    """
    with filepath.open("w") as f:
        yaml.safe_dump(data, f)


def _get_atomic_mass(symbol: str) -> float:
    """Helper to get atomic mass with caching."""
    if symbol not in _ATOMIC_MASSES_CACHE:
        from ase.data import atomic_masses, atomic_numbers
        _ATOMIC_MASSES_CACHE[symbol] = atomic_masses[atomic_numbers[symbol]]
    return _ATOMIC_MASSES_CACHE[symbol]


import re
from typing import TextIO


class LAMMPSWriteError(Exception):
    """Exception raised for errors during LAMMPS streaming writes."""

def write_lammps_streaming(
    fileobj: TextIO,
    atoms: Atoms,
    species: list[str],
    atom_style: str = "atomic"
) -> None:
    """
    Writes a single frame in LAMMPS data format to an open file object.
    Optimized for streaming large trajectories using minimal memory and vectorized formatting.

    Args:
        fileobj: An open file object (in write mode).
        atoms: The ASE Atoms object to write.
        species: List of chemical symbols mapping to types 1..N.
        atom_style: LAMMPS atom style (currently only 'atomic' supported for streaming).
    """
    if not hasattr(fileobj, 'write') or not callable(fileobj.write):
        raise TypeError("fileobj must be a writable file-like object")

    # Validate species names are safe (prevent injection)
    safe_symbol_regex = re.compile(r"^[A-Za-z]+$")
    for s in species:
        if not safe_symbol_regex.match(s):
            raise ValueError(f"Invalid chemical symbol in species list: {s}")

    natoms = len(atoms)
    if natoms == 0:
        raise ValueError("Cannot write an empty atoms object.")

    # Validate that all atoms exist in species list before starting write
    symbols = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]
    unique_symbols = set(symbols)
    species_set = set(species)
    missing = unique_symbols - species_set
    if missing:
        raise ValueError(f"Symbols {missing} not in provided species list: {species}")

    try:
        # 1. Header
        fileobj.write("LAMMPS data file via pyacemaker streaming\n\n")
        fileobj.write(f"{natoms} atoms\n")
        fileobj.write(f"{len(species)} atom types\n\n")

        # 2. Box (Support for non-orthogonal cells)
        cell = atoms.get_cell() # type: ignore[no-untyped-call]

        # In LAMMPS:
        # a = xhi - xlo
        # b = sqrt((yhi-ylo)^2 + xy^2)
        # c = sqrt((zhi-zlo)^2 + xz^2 + yz^2)
        xhi = cell[0, 0]
        xy = cell[1, 0]
        yhi = cell[1, 1]
        xz = cell[2, 0]
        yz = cell[2, 1]
        zhi = cell[2, 2]

        xlo, ylo, zlo = 0.0, 0.0, 0.0

        # Handle tilt factors if non-orthogonal
        is_orthogonal = np.allclose(cell, np.diag(np.diag(cell)))

        if is_orthogonal:
            fileobj.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
            fileobj.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
            fileobj.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        else:
            # Bound calculation (ASE standard for triclinic)
            xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
            xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
            ylo_bound = ylo + min(0.0, yz)
            yhi_bound = yhi + max(0.0, yz)
            zlo_bound = zlo
            zhi_bound = zhi

            fileobj.write(f"{xlo_bound:.6f} {xhi_bound:.6f} xlo xhi\n")
            fileobj.write(f"{ylo_bound:.6f} {yhi_bound:.6f} ylo yhi\n")
            fileobj.write(f"{zlo_bound:.6f} {zhi_bound:.6f} zlo zhi\n")
            fileobj.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n\n")

        # 3. Masses
        fileobj.write("Masses\n\n")

        # Create a mapping from symbol to type ID (1-based)
        type_map = {s: i + 1 for i, s in enumerate(species)}

        for s in species:
            type_id = type_map[s]
            mass = _get_atomic_mass(s)
            fileobj.write(f"{type_id} {mass:.4f} # {s}\n")

        fileobj.write("\n")

        # 4. Atoms
        fileobj.write("Atoms # atomic\n\n")

        pos = atoms.get_positions() # type: ignore[no-untyped-call]

        def line_generator() -> Iterable[str]:
            for i in range(natoms):
                s = symbols[i]
                t = type_map[s]
                # 1-based index
                yield f"{i+1} {t} {pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f}\n"

        fileobj.writelines(line_generator())
        fileobj.write("\n")
    except Exception as e:
        raise LAMMPSWriteError(f"I/O Error writing LAMMPS format: {e}") from e
