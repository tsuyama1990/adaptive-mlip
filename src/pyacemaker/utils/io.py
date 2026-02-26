from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read


def load_config(filepath: Path) -> dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Dictionary containing configuration.
    """
    with filepath.open("r") as f:
        # Use safe_load for security
        return yaml.safe_load(f) or {}


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
    from ase.io import iread

    try:
        # iread returns a generator
        for i, atoms in enumerate(iread(str(data_path))):
            if i >= max_frames:
                break
            symbols.update(atoms.get_chemical_symbols())
    except Exception:
        # Fallback for single frame reading if iread fails or not supported for format
        try:
            atoms = read(str(data_path), index=0)
            if isinstance(atoms, list):
                atoms = atoms[0]
            if isinstance(atoms, Atoms):
                symbols.update(atoms.get_chemical_symbols())
        except Exception:
            pass

    return sorted(list(symbols))


def dump_yaml(data: Any, filepath: Path) -> None:
    """
    Dumps data to a YAML file safely.

    Args:
        data: The data to dump (dict, list, etc).
        filepath: Path to the output file.
    """
    with filepath.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


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
        msg = "Only orthogonal simulation boxes are supported by streaming writer."
        raise ValueError(msg)

    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = cell[0,0], cell[1,1], cell[2,2]

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

        # Validation for missing species
        if sym not in type_map:
             msg = f"Atom {atom_id} has symbol {sym} not found in species list {elements}"
             raise ValueError(msg)

        typ = type_map[sym]
        pos = positions[i]
        fileobj.write(f"{atom_id} {typ} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
