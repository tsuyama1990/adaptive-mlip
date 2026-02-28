import logging
from pathlib import Path
from typing import Any, TextIO

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
        msg = f"Configuration file not found: {filepath}"
        raise FileNotFoundError(msg)

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
        # Ensure streaming by iterating directly without converting to list.
        gen = iread(str(data_path), index=f":{max_frames}")
        for atoms in gen:
            if isinstance(atoms, Atoms):
                new_syms = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
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
    fileobj: TextIO, atoms: Atoms, species: list[str], atom_style: str = "atomic"
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

    from pyacemaker.domain_models.constants import LAMMPS_FORMAT_STREAMING_HEADER

    # 1. Header
    fileobj.write(LAMMPS_FORMAT_STREAMING_HEADER)
    fileobj.write(f"{natoms} atoms\n")
    fileobj.write(f"{len(species)} atom types\n\n")

    # 2. Box (Support triclinic cells safely)
    cell = atoms.get_cell()  # type: ignore[no-untyped-call]
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi = cell[0, 0]
    yhi = cell[1, 1]
    zhi = cell[2, 2]
    xy, xz, yz = cell[0, 1], cell[0, 2], cell[1, 2]

    # In LAMMPS, xhi is bound by xhi-xlo, not max(x). But we assume origin at 0,0,0
    fileobj.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
    fileobj.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
    fileobj.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")

    if abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6:
        fileobj.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")

    fileobj.write("\n")

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
    # Avoid getting the entire positions array into memory if possible by using atoms.positions memory view.
    # We write directly to the file inside the loop.

    pos = atoms.positions  # memoryview / reference to existing array
    symbols = atoms.symbols  # property returning a view-like object

    # Write chunks of lines directly to avoid huge string or writelines buffering
    for i in range(natoms):
        s = symbols[i]
        try:
            t = type_map[s]
        except KeyError as err:
            msg = f"Symbol {s} not in provided species list: {species}"
            raise KeyError(msg) from err

        # 1-based index
        fileobj.write(f"{i + 1} {t} {pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f}\n")

    fileobj.write("\n")

def _parse_and_write_lattice(props_line: str, output_fileobj: TextIO) -> None:
    import re
    lattice_match = re.search(r'Lattice="([^"]+)"', props_line)
    if lattice_match:
        try:
            l_vals = [float(x) for x in lattice_match.group(1).split()]
            if len(l_vals) == 9:
                xhi = l_vals[0]
                xy = l_vals[1]
                xz = l_vals[2]
                yhi = l_vals[4]
                yz = l_vals[5]
                zhi = l_vals[8]

                output_fileobj.write(f"0.000000 {xhi:.6f} xlo xhi\n")
                output_fileobj.write(f"0.000000 {yhi:.6f} ylo yhi\n")
                output_fileobj.write(f"0.000000 {zhi:.6f} zlo zhi\n")

                if abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6:
                    output_fileobj.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")

                output_fileobj.write("\n")
        except (ValueError, IndexError):
            pass

def _write_masses(output_fileobj: TextIO, species: list[str], type_map: dict[str, int]) -> None:
    output_fileobj.write("Masses\n\n")
    for s in species:
        type_id = type_map[s]
        mass = _get_atomic_mass(s)
        output_fileobj.write(f"{type_id} {mass:.4f} # {s}\n")
    output_fileobj.write("\n")

def _write_atoms(fin: TextIO, output_fileobj: TextIO, natoms: int, species: list[str], type_map: dict[str, int]) -> None:
    output_fileobj.write("Atoms # atomic\n\n")
    for i in range(natoms):
        line = fin.readline()
        if not line:
            break
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0]
            try:
                t = type_map[sym]
            except KeyError as err:
                msg = f"Symbol {sym} not in provided species list: {species}"
                raise KeyError(msg) from err
            x, y, z = parts[1], parts[2], parts[3]
            output_fileobj.write(f"{i + 1} {t} {x} {y} {z}\n")

def _read_natoms(first_line: str, input_path: Path) -> int:
    if not first_line:
        msg = f"Input structure file {input_path} is empty."
        raise ValueError(msg)
    try:
        return int(first_line.strip())
    except ValueError as err:
        msg = f"Invalid extxyz format, expected integer for atom count on line 1, got: {first_line}"
        raise ValueError(msg) from err

def _write_header(output_fileobj: TextIO, natoms: int, num_species: int) -> None:
    from pyacemaker.domain_models.constants import LAMMPS_FORMAT_STREAMING_HEADER
    output_fileobj.write(LAMMPS_FORMAT_STREAMING_HEADER)
    output_fileobj.write(f"{natoms} atoms\n")
    output_fileobj.write(f"{num_species} atom types\n\n")

def stream_extxyz_to_lammps(input_path: Path, output_fileobj: TextIO, species: list[str]) -> list[str]:
    """
    Streams an extended XYZ file directly to LAMMPS data format.
    Guarantees O(1) memory usage by never materializing ASE Atoms objects.
    """
    with input_path.open("r") as fin:
        first_line = fin.readline()
        natoms = _read_natoms(first_line, input_path)

        props_line = fin.readline().strip()

        _write_header(output_fileobj, natoms, len(species))

        _parse_and_write_lattice(props_line, output_fileobj)

        type_map = {s: i + 1 for i, s in enumerate(species)}
        _write_masses(output_fileobj, species, type_map)

        _write_atoms(fin, output_fileobj, natoms, species, type_map)

    return species
