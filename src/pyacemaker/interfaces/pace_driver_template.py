PACE_DRIVER_TEMPLATE = """
import sys
import os
import numpy as np
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS
from pydantic import BaseModel, Field, ValidationError

class PaceDriverConfig(BaseModel):
    potential_path: str = Field(..., pattern=r"^[^;&|`$<>]+$", description="Path to the potential file")

def read_input():
    try:
        lines = sys.stdin.readlines()
        if not lines:
            return None, None, None

        num_atoms = int(lines[0].strip())

        cell = np.zeros((3, 3))
        cell[0] = [float(x) for x in lines[1].split()]
        cell[1] = [float(x) for x in lines[2].split()]
        cell[2] = [float(x) for x in lines[3].split()]

        coords = []
        for i in range(num_atoms):
            coords.append([float(x) for x in lines[4+i].split()])

        return num_atoms, cell, np.array(coords)
    except Exception as e:
        sys.stderr.write(f"Error reading input: {e}\n")
        sys.exit(1)

def main():
    try:
        env_potential = os.environ.get("PACE_POTENTIAL_PATH")
        if not env_potential:
            sys.stderr.write("Error: PACE_POTENTIAL_PATH not set\n")
            sys.exit(1)

        try:
            config = PaceDriverConfig(potential_path=env_potential)
        except ValidationError:
            sys.stderr.write("Error: Invalid characters in potential path\n")
            sys.exit(1)

        if not os.path.exists(config.potential_path):
            sys.stderr.write(f"Error: Potential file not found at {config.potential_path}\n")
            sys.exit(1)

        if not os.path.exists("pos.con"):
            sys.stderr.write("Error: pos.con not found\n")
            sys.exit(1)

        try:
            template = read("pos.con", format="eon")
        except Exception:
            template = read("pos.con")

        n, cell, coords = read_input()
        if n is None:
            sys.exit(0)

        if n != len(template):
            sys.stderr.write(f"Error: Atom count mismatch ({n} vs {len(template)})\n")
            sys.exit(1)

        template.set_cell(cell)
        template.set_positions(coords)

        species = sorted(list(set(template.get_chemical_symbols())))
        species_str = " ".join(species)

        parameters = {
            "pair_style": "pace",
            "pair_coeff": [f"* * {config.potential_path} {species_str}"]
        }

        calc = LAMMPS(parameters=parameters, files=[config.potential_path])
        template.calc = calc

        energy = template.get_potential_energy()
        forces = template.get_forces()

        print(f"{energy:.16f}")
        for f in forces:
            print(f"{f[0]:.16f} {f[1]:.16f} {f[2]:.16f}")

    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
