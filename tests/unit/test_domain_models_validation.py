import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult


def test_validation_config_defaults() -> None:
    config = ValidationConfig()
    assert config.phonon_supercell == [2, 2, 2]
    assert config.phonon_displacement == 0.01
    assert config.elastic_strain == 0.01
    assert config.imaginary_frequency_tolerance == -0.05
    assert config.symprec == 1e-5


def test_validation_config_custom_valid() -> None:
    config = ValidationConfig(
        phonon_supercell=[3, 3, 3],
        phonon_displacement=0.02,
        elastic_strain=0.005,
        imaginary_frequency_tolerance=-0.1,
        symprec=1e-4,
    )
    assert config.phonon_supercell == [3, 3, 3]
    assert config.phonon_displacement == 0.02
    assert config.elastic_strain == 0.005
    assert config.imaginary_frequency_tolerance == -0.1
    assert config.symprec == 1e-4


def test_validation_config_invalid() -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(phonon_displacement=-0.1)  # Must be positive
    with pytest.raises(ValidationError):
        ValidationConfig(elastic_strain=-0.01)  # Must be positive
    with pytest.raises(ValidationError):
        ValidationConfig(symprec=-1e-5)  # Must be positive


def test_validation_result_valid() -> None:
    result = ValidationResult(
        phonon_stable=True,
        elastic_stable=False,
        imaginary_frequencies=[],
        elastic_tensor=[[1.0, 0.0], [0.0, 1.0]],
        bulk_modulus=100.0,
        shear_modulus=50.0,
        plots={"band": "base64string"},
    )
    assert result.phonon_stable is True
    assert result.elastic_stable is False
    assert result.bulk_modulus == 100.0
    assert result.plots
    assert result.plots["band"] == "base64string"


def test_validation_result_missing_fields() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(
            phonon_stable=True,
            # Missing other required fields
        )  # type: ignore[call-arg]
