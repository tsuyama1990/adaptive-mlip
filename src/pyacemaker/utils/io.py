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
    symbols = set()
    try:
        # Optimization: Use iread to peek. Stop if we have 'enough' frames or symbols stabilize?
        # Difficult to know if symbols stabilize. Just read max_frames.
        gen = iread(str(data_path), index=f":{max_frames}")
        for atoms in gen:
            if isinstance(atoms, Atoms):
                new_syms = set(atoms.get_chemical_symbols())
                # If we found new symbols, update.
                if not new_syms.issubset(symbols):
                    symbols.update(new_syms)
    except Exception:
        logger.warning(
            f"Could not fully read {data_path} to detect elements. Elements detected so far: {symbols}"
        )

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


def write_lammps_streaming(
    fileobj: Any, atoms: Atoms, species: list[str], atom_style: str = "atomic"
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
    natoms = len(atoms)

    # 1. Header
    fileobj.write("LAMMPS data file via pyacemaker streaming\n\n")
    fileobj.write(f"{natoms} atoms\n")
    fileobj.write(f"{len(species)} atom types\n\n")

    # 2. Box
    cell = atoms.get_cell()
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError("Streaming write currently only supports orthogonal cells")

    xlo, xhi = 0.0, cell[0, 0]
    ylo, yhi = 0.0, cell[1, 1]
    zlo, zhi = 0.0, cell[2, 2]

    fileobj.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
    fileobj.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
    fileobj.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

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

    # Optimize Atom Writing:
    # Use direct array access and iterators to avoid creating large intermediate lists/arrays if possible.
    # But atoms.get_positions() returns a copy anyway.

    pos = atoms.get_positions()  # (N, 3)
    symbols = atoms.get_chemical_symbols()  # List of strings (N)

    # Generator for lines to keep memory usage O(1) per line (after pos array overhead)
    # This avoids creating a huge string buffer or list of strings.
    def line_generator() -> Iterable[str]:
        for i in range(natoms):
            s = symbols[i]
            try:
                t = type_map[s]
            except KeyError:
                raise KeyError(f"Symbol {s} not in provided species list: {species}")

            # 1-based index
            yield f"{i + 1} {t} {pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f}\n"

    fileobj.writelines(line_generator())

    fileobj.write("\n")
