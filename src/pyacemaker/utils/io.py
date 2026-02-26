import json
import logging
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
        # Use iread to peek without loading whole file
        gen = iread(str(data_path), index=f":{max_frames}")
        for atoms in gen:
            if isinstance(atoms, Atoms):
                symbols.update(atoms.get_chemical_symbols())
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


def write_lammps_streaming(
    fileobj: Any,
    atoms: Atoms,
    species: list[str],
    atom_style: str = "atomic"
) -> None:
    """
    Writes a single frame in LAMMPS data format to an open file object.
    Optimized for streaming large trajectories.

    Args:
        fileobj: An open file object (in write mode).
        atoms: The ASE Atoms object to write.
        species: List of chemical symbols mapping to types 1..N.
        atom_style: LAMMPS atom style (currently only 'atomic' supported for streaming).
    """
    # Use direct array access for speed
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    symbols = atoms.get_chemical_symbols()

    # Determine types
    # Pre-compute map
    type_map = {s: i + 1 for i, s in enumerate(species)}

    natoms = len(atoms)

    # 1. Header
    fileobj.write("LAMMPS data file via pyacemaker streaming\n\n")
    fileobj.write(f"{natoms} atoms\n")
    fileobj.write(f"{len(species)} atom types\n\n")

    # 2. Box
    # Assumes orthogonal box for simplicity in this streaming version
    # If triclinic, need xy xz yz
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

    for s in species:
        type_id = type_map[s]
        mass = _get_atomic_mass(s)
        fileobj.write(f"{type_id} {mass:.4f} # {s}\n")

    fileobj.write("\n")

    # 4. Atoms
    fileobj.write("Atoms # atomic\n\n")

    # Iterate and write lines
    # Format: id type x y z
    for i in range(natoms):
        s = symbols[i]
        try:
             t = type_map[s]
        except KeyError as e:
             raise KeyError(f"Symbol {s} not in provided species list: {species}") from e

        x, y, z = pos[i]
        # id is 1-based
        fileobj.write(f"{i+1} {t} {x:.6f} {y:.6f} {z:.6f}\n")

    fileobj.write("\n")
