from ase import Atoms


def get_species_order(atoms: Atoms) -> list[str]:
    """
    Returns the sorted list of unique chemical symbols in the structure.
    Used for determining LAMMPS atom types.

    Args:
        atoms: ASE Atoms object.

    Returns:
        List of unique chemical symbols (e.g., ["Al", "Ni"]).

    Raises:
        TypeError: If atoms is not an ASE Atoms object.
    """
    if not isinstance(atoms, Atoms):
        msg = f"Expected ASE Atoms object, got {type(atoms)}."
        raise TypeError(msg)

    return sorted({atom.symbol for atom in atoms})
