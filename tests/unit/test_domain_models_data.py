import pytest
from ase import Atoms
from pydantic import ValidationError

from pyacemaker.domain_models.data import AtomStructure


def test_atom_structure_valid():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    structure = AtomStructure(
        atoms=atoms,
        provenance="DIRECT_SAMPLING",
        energy=-13.6
    )
    assert structure.atoms == atoms
    assert structure.energy == -13.6
    assert structure.provenance == "DIRECT_SAMPLING"

def test_atom_structure_invalid():
    # Missing required fields
    with pytest.raises(ValidationError):
        AtomStructure(provenance="DIRECT_SAMPLING")

def test_atom_structure_metadata():
    atoms = Atoms("He")
    structure = AtomStructure(
        atoms=atoms,
        provenance="TEST",
        metadata={"temp": 300}
    )
    assert structure.metadata["temp"] == 300
