from .defaults import (
    DANGEROUS_PATH_CHARS,
    DEFAULT_LJ_PARAMS,
    DEFAULT_MD_MINIMIZE_FTOL,
    DEFAULT_MD_MINIMIZE_TOL,
    DEFAULT_RAM_DISK_PATH,
    DEFAULT_STRAIN_RANGE,
    EMBEDDING_TOLERANCE_CELL,
    ERR_GEN_BASE_FAIL,
    ERR_GEN_NCAND_NEG,
    ERR_M3GNET_PRED_FAIL,
    ERR_ORACLE_FAILED,
    ERR_ORACLE_ITERATOR,
    ERR_POTENTIAL_NOT_FOUND,
    ERR_SIM_EXEC_FAIL,
    ERR_SIM_SECURITY_FAIL,
    ERR_SIM_SETUP_FAIL,
    ERR_SIM_UNEXPECTED,
    ERR_STRUCTURE_NONE,
    ERR_VAL_POT_NONE,
    ERR_VAL_POT_NOT_FILE,
    ERR_VAL_POT_OUTSIDE,
    ERR_VAL_REQ_STRUCT,
    ERR_VAL_STRUCT_EMPTY,
    ERR_VAL_STRUCT_NONE,
    ERR_VAL_STRUCT_TYPE,
    FALLBACK_LJ_PARAMS,
    LAMMPS_MIN_STYLE_CG,
    LAMMPS_MINIMIZE_MAX_ITER,
    LAMMPS_MINIMIZE_STEPS,
    LAMMPS_SAFE_CMD_PATTERN,
    LAMMPS_SCREEN_ARG,
    LAMMPS_VELOCITY_SEED,
    RECIPROCAL_FACTOR,
)

# Default random seed for EON simulations
DEFAULT_EON_SEED = 12345

# Default EON executable name
DEFAULT_EON_EXECUTABLE = "eonclient"

# PACE Driver script template (uses environment variables for security)
PACE_DRIVER_TEMPLATE = """
import sys
import os
import numpy as np
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS

# Read potential path from environment for security
POTENTIAL_PATH = os.environ.get("PACE_POTENTIAL_PATH")
if not POTENTIAL_PATH:
    sys.stderr.write("Error: PACE_POTENTIAL_PATH not set\\n")
    sys.exit(1)

if not os.path.exists(POTENTIAL_PATH):
    sys.stderr.write(f"Error: Potential file not found at {POTENTIAL_PATH}\\n")
    sys.exit(1)

def read_input():
    try:
        lines = sys.stdin.readlines()
        if not lines:
            return None, None, None

        # Parse EON format (assumed based on standard command line potentials)
        # 1. Number of atoms
        num_atoms = int(lines[0].strip())

        # 2. Box (3 lines)
        cell = np.zeros((3, 3))
        cell[0] = [float(x) for x in lines[1].split()]
        cell[1] = [float(x) for x in lines[2].split()]
        cell[2] = [float(x) for x in lines[3].split()]

        # 3. Coordinates (N lines)
        # EON usually sends only coordinates x y z, not species.
        # We need to map them to species from a template.
        coords = []
        for i in range(num_atoms):
            coords.append([float(x) for x in lines[4+i].split()])

        return num_atoms, cell, np.array(coords)
    except Exception as e:
        sys.stderr.write(f"Error reading input: {e}\\n")
        sys.exit(1)

def main():
    try:
        # 1. Load template structure (pos.con) to get species
        if not os.path.exists("pos.con"):
            sys.stderr.write("Error: pos.con not found\\n")
            sys.exit(1)

        try:
            # EON uses .con format, ASE supports it
            template = read("pos.con", format="eon")
        except Exception:
            # Fallback if ASE cannot read eon format directly or extension issue
            # Try reading as 'con' or just assume it works if supported
            template = read("pos.con")

        # 2. Read current configuration from stdin
        n, cell, coords = read_input()
        if n is None:
            sys.exit(0)

        if n != len(template):
            sys.stderr.write(f"Error: Atom count mismatch ({n} vs {len(template)})\\n")
            sys.exit(1)

        # 3. Update structure
        template.set_cell(cell)
        template.set_positions(coords)

        # 4. Setup Calculator
        # Using LAMMPS via ASE is robust.
        # For speed, one might use lammps python module directly,
        # but constructing the input script for PACE is complex.
        # We rely on ASE's LAMMPS calculator or similar.
        # We need to specify the potential file.

        # We construct a pair_style command for PACE.
        # Assuming 'pace' pair style is available in the LAMMPS binary.
        # pair_style pace
        # pair_coeff * * potential.yace Element1 Element2 ...

        species = sorted(list(set(template.get_chemical_symbols())))
        species_str = " ".join(species)

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {POTENTIAL_PATH} {species_str}"]
        }

        calc = LAMMPS(parameters=parameters, files=[POTENTIAL_PATH])
        template.calc = calc

        # 5. Compute
        energy = template.get_potential_energy()
        forces = template.get_forces()

        # 6. Output results
        # Format: Energy (1 line)
        # Forces (N lines, x y z)
        print(f"{energy:.16f}")
        for f in forces:
            print(f"{f[0]:.16f} {f[1]:.16f} {f[2]:.16f}")

    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

__all__ = [
    "DANGEROUS_PATH_CHARS",
    "DEFAULT_EON_EXECUTABLE",
    "DEFAULT_EON_SEED",
    "DEFAULT_LJ_PARAMS",
    "DEFAULT_MD_MINIMIZE_FTOL",
    "DEFAULT_MD_MINIMIZE_TOL",
    "DEFAULT_RAM_DISK_PATH",
    "DEFAULT_STRAIN_RANGE",
    "EMBEDDING_TOLERANCE_CELL",
    "ERR_GEN_BASE_FAIL",
    "ERR_GEN_NCAND_NEG",
    "ERR_M3GNET_PRED_FAIL",
    "ERR_ORACLE_FAILED",
    "ERR_ORACLE_ITERATOR",
    "ERR_POTENTIAL_NOT_FOUND",
    "ERR_SIM_EXEC_FAIL",
    "ERR_SIM_SECURITY_FAIL",
    "ERR_SIM_SETUP_FAIL",
    "ERR_SIM_UNEXPECTED",
    "ERR_STRUCTURE_NONE",
    "ERR_VAL_POT_NONE",
    "ERR_VAL_POT_NOT_FILE",
    "ERR_VAL_POT_OUTSIDE",
    "ERR_VAL_REQ_STRUCT",
    "ERR_VAL_STRUCT_EMPTY",
    "ERR_VAL_STRUCT_NONE",
    "ERR_VAL_STRUCT_TYPE",
    "FALLBACK_LJ_PARAMS",
    "LAMMPS_MINIMIZE_MAX_ITER",
    "LAMMPS_MINIMIZE_STEPS",
    "LAMMPS_MIN_STYLE_CG",
    "LAMMPS_SAFE_CMD_PATTERN",
    "LAMMPS_SCREEN_ARG",
    "LAMMPS_VELOCITY_SEED",
    "PACE_DRIVER_TEMPLATE",
    "RECIPROCAL_FACTOR",
]
