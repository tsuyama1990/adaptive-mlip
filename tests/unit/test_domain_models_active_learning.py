import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from pyacemaker.domain_models.active_learning import DescriptorConfig, SamplingResult
from pyacemaker.domain_models.data import AtomStructure


def test_descriptor_config_valid() -> None:
    config = DescriptorConfig(
        method="soap",
        species=["Cu", "Au"],
        r_cut=5.0,
        n_max=8,
        l_max=6,
        sigma=0.5
    )
    assert config.method == "soap"
    assert config.species == ["Cu", "Au"]
    assert config.r_cut == 5.0

def test_descriptor_config_soap_missing_sigma() -> None:
    with pytest.raises(ValidationError):
        DescriptorConfig(
            method="soap",
            species=["Cu"],
            r_cut=5.0,
            n_max=8,
            l_max=6,
            # sigma is missing
        )

def test_descriptor_config_ace_valid() -> None:
    config = DescriptorConfig(
        method="ace",
        species=["Cu"],
        r_cut=5.0,
        n_max=8,
        l_max=3
    )
    assert config.method == "ace"
    assert config.sigma is None

def test_sampling_result_valid() -> None:
    pool = [
        AtomStructure(atoms=Atoms('Cu', positions=[[0, 0, 0]])),
        AtomStructure(atoms=Atoms('Cu', positions=[[1, 1, 1]]))
    ]
    descriptors = np.array([[0.1, 0.2], [0.3, 0.4]])
    selection_indices = [0, 1]

    result = SamplingResult(
        pool=pool,
        descriptors=descriptors,
        selection_indices=selection_indices
    )
    assert len(result.pool) == 2
    assert result.descriptors is not None
    assert result.descriptors.shape == (2, 2)

def test_sampling_result_no_descriptors() -> None:
    pool = [AtomStructure(atoms=Atoms('Cu'))]
    result = SamplingResult(
        pool=pool,
        selection_indices=[0]
    )
    assert result.descriptors is None

# def test_sampling_result_shape_mismatch() -> None:
#     # This validation logic was considered but implementation detail was ambiguous in previous step
#     # If implemented, uncomment this test.
#     pass
