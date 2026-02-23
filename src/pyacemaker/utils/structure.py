from ase import Atoms


def get_species_order(atoms: Atoms) -> list[str]:
    """
    Returns the sorted list of unique chemical symbols in the structure.
    Used for determining LAMMPS atom types.

    Args:
        atoms: ASE Atoms object.

    Returns:
        List of unique chemical symbols (e.g., ["Al", "Ni"]).
    """
    return sorted({atom.symbol for atom in atoms})
