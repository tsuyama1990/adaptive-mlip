from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.validator import Validator
from pyacemaker.domain_models.validation import ValidationConfig


@pytest.fixture
def validator_dependencies():
    return {
        "phonon": MagicMock(),
        "elastic": MagicMock(),
        "report": MagicMock()
    }

@pytest.fixture
def validator(validator_dependencies):
    config = ValidationConfig()
    return Validator(
        config,
        validator_dependencies["phonon"],
        validator_dependencies["elastic"],
        validator_dependencies["report"]
    )

def test_uat_07_01_validate_potential_pass(validator, validator_dependencies):
    """Scenario 07-01: 'Validate Potential' (PASS)"""
    # 1. Preparation
    potential_path = Path("test_potential.yace")
    potential_path.touch()
    report_path = Path("validation_report.html")

    # Mock successful validation
    validator_dependencies["phonon"].check_stability.return_value = (True, "base64_phonon")
    validator_dependencies["elastic"].calculate_properties.return_value = (
        True, {"C11": 200.0, "C12": 100.0, "C44": 50.0}, 150.0, "base64_elastic"
    )

    # 2. Action
    structure = Atoms("Fe", positions=[[0,0,0]], cell=[2.8,2.8,2.8])

    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = structure
        result = validator.validate(potential_path, report_path, structure=structure)

    # 3. Expectation
    assert result.phonon_stable is True
    assert result.elastic_stable is True
    assert "C11" in result.c_ij
    assert result.report_path == str(report_path)

    validator_dependencies["report"].generate.assert_called_once()
    validator_dependencies["report"].save.assert_called_once_with(report_path, validator_dependencies["report"].generate.return_value)

    if potential_path.exists():
        potential_path.unlink()

def test_uat_07_02_unstable_detection(validator, validator_dependencies):
    """Scenario 07-02: 'Unstable Detection'"""
    # 1. Preparation
    potential_path = Path("test_unstable.yace")
    potential_path.touch()
    report_path = Path("validation_unstable_report.html")

    # Mock unstable (phonon unstable)
    validator_dependencies["phonon"].check_stability.return_value = (False, "base64_phonon_unstable")
    validator_dependencies["elastic"].calculate_properties.return_value = (
        True, {"C11": 200.0}, 150.0, "base64_elastic"
    )

    # 2. Action
    structure = Atoms("Fe", positions=[[0,0,0]], cell=[2.8,2.8,2.8])

    with patch.object(validator, "_relax_structure") as mock_relax:
        mock_relax.return_value = structure
        result = validator.validate(potential_path, report_path, structure=structure)

    # 3. Expectation
    assert result.phonon_stable is False
    # Report should still be generated
    validator_dependencies["report"].generate.assert_called_once()

    if potential_path.exists():
        potential_path.unlink()
