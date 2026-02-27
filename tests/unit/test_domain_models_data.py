import numpy as np
from ase import Atoms

from pyacemaker.domain_models.data import AtomStructure


def test_atom_structure_initialization() -> None:
    """Test basic initialization and validation."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Valid initialization
    structure = AtomStructure(atoms=atoms)
    assert structure.energy is None
    assert structure.forces is None

    # Valid initialization with optional fields
    forces = np.zeros((3, 3))
    structure = AtomStructure(
        atoms=atoms,
        energy=-10.5,
        forces=forces,
        stress=np.zeros(6),
        uncertainty=0.1,
        provenance={"step": "test"}
    )
    assert structure.energy == -10.5
    assert np.array_equal(structure.forces, forces)
    assert structure.provenance["step"] == "test"

def test_atom_structure_to_ase() -> None:
    """Test conversion back to ASE Atoms."""
    atoms = Atoms("H", positions=[[0, 0, 0]])
    structure = AtomStructure(
        atoms=atoms,
        energy=-5.0,
        forces=np.array([[0.1, 0.2, 0.3]]),
        uncertainty=0.05,
        provenance={"method": "dft"}
    )

    ase_atoms = structure.to_ase()

    # Verify metadata transfer
    assert ase_atoms.info["energy"] == -5.0
    assert ase_atoms.info["uncertainty"] == 0.05
    assert ase_atoms.info["gamma"] == 0.05
    assert ase_atoms.info["provenance_method"] == "dft"
    assert np.allclose(ase_atoms.arrays["forces"], [[0.1, 0.2, 0.3]])

def test_atom_structure_from_ase() -> None:
    """Test creation from ASE Atoms."""
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.info["energy"] = -3.0
    atoms.info["uncertainty"] = 0.2
    atoms.info["provenance_step"] = "init"
    atoms.arrays["forces"] = np.array([[1.0, 0.0, 0.0]])

    structure = AtomStructure.from_ase(atoms)

    assert structure.energy == -3.0
    assert structure.uncertainty == 0.2
    assert structure.provenance["step"] == "init"
    assert np.allclose(structure.forces, [[1.0, 0.0, 0.0]])

def test_atom_structure_from_ase_calculator() -> None:
    """Test creation from ASE Atoms with Calculator attached."""
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = Atoms("H", positions=[[0, 0, 0]])
    calc = SinglePointCalculator(
        atoms,
        energy=-4.0,
        forces=[[0.5, 0.5, 0.5]],
        stress=[0]*6
    )
    atoms.calc = calc

    structure = AtomStructure.from_ase(atoms)

    assert structure.energy == -4.0
    assert np.allclose(structure.forces, [[0.5, 0.5, 0.5]])
    assert np.allclose(structure.stress, [0]*6)
