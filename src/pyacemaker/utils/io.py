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
    Optimized for streaming large trajectories using vectorized operations.

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
    # species list index i -> type i+1
    type_map = {s: i + 1 for i, s in enumerate(species)}

    for s in species:
        type_id = type_map[s]
        mass = _get_atomic_mass(s)
        fileobj.write(f"{type_id} {mass:.4f} # {s}\n")

    fileobj.write("\n")

    # 4. Atoms
    fileobj.write("Atoms # atomic\n\n")

    # Optimize Atom Writing: Vectorize
    # Get arrays
    pos = atoms.get_positions() # (N, 3)
    symbols = np.array(atoms.get_chemical_symbols()) # (N,)

    # Map symbols to types using lookup (vectorized if possible, or list comp)
    # A fast way is using numpy extract/place or just list comprehension which is fast enough for 1M items compared to I/O
    # But np.vectorize with a dict lookup is okay.
    # Better: Precompute integer types from atomic numbers if possible?
    # species list is small.

    # Create an integer array for types
    # Initialize with 0
    atom_types = np.zeros(natoms, dtype=int)

    # Fill atom_types
    for s, t_id in type_map.items():
        # Mask where symbol matches s
        mask = (symbols == s)
        atom_types[mask] = t_id

    # Check for unassigned (missing species)
    if np.any(atom_types == 0):
        # Find first missing
        missing_mask = (atom_types == 0)
        missing_sym = symbols[missing_mask][0]
        raise KeyError(f"Symbol {missing_sym} not in provided species list: {species}")

    # Construct the data array to dump
    # Columns: id type x y z
    ids = np.arange(1, natoms + 1)

    # We can write line by line or construct a big string buffer.
    # For OOM safety on massive systems, line by line or chunks is better.
    # np.savetxt is fast but formatting mixed int/float is tricky.

    # Manual loop with f-string is usually the bottleneck in python.
    # Let's use np.savetxt with a structured array or just loop over the arrays which is faster than getting atom by atom.

    # Iterate over zip of arrays is faster than atoms indexing
    for i, (atom_id, type_id, (x, y, z)) in enumerate(zip(ids, atom_types, pos)):
        fileobj.write(f"{atom_id} {type_id} {x:.6f} {y:.6f} {z:.6f}\n")

    fileobj.write("\n")
