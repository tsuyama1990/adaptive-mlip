import pytest

from pyacemaker.domain_models.validation import ValidationConfig, ValidationResult


def test_validation_config_defaults():
    config = ValidationConfig()
    assert config.phonon_supercell == [2, 2, 2]
    assert config.phonon_displacement == 0.01
    assert config.phonon_imaginary_tol == -0.05
    assert config.elastic_strain == 0.01
    assert config.elastic_steps == 5

def test_validation_config_custom():
    config = ValidationConfig(
        phonon_supercell=[3, 3, 3],
        phonon_displacement=0.02,
        phonon_imaginary_tol=-0.1,
        elastic_strain=0.02,
        elastic_steps=7
    )
    assert config.phonon_supercell == [3, 3, 3]
    assert config.phonon_displacement == 0.02
    assert config.phonon_imaginary_tol == -0.1
    assert config.elastic_strain == 0.02
    assert config.elastic_steps == 7

def test_validation_result_valid():
    result = ValidationResult(
        phonon_stable=True,
        elastic_stable=True,
        c_ij={"C11": 200.0, "C12": 100.0, "C44": 50.0},
        bulk_modulus=150.0,
        report_path="report.html"
    )
    assert result.phonon_stable is True
    assert result.elastic_stable is True
    assert result.c_ij["C11"] == 200.0

def test_validation_result_missing_fields():
    with pytest.raises(ValueError, match="Field required"):
        ValidationResult(phonon_stable=True) # Missing required fields
