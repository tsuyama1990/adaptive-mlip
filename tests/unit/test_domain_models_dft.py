import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import DFTConfig


def test_dft_config_full_valid() -> None:
    """Test full initialization of DFTConfig with all optional fields."""
    config = DFTConfig(
        code="quantum_espresso",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.5,
        smearing_type="gaussian",
        smearing_width=0.05,
        diagonalization="cg",
        pseudopotentials={"Fe": "Fe_pseudo.UPF"},
    )
    assert config.mixing_beta == 0.5
    assert config.smearing_type == "gaussian"
    assert config.smearing_width == 0.05
    assert config.diagonalization == "cg"
    assert config.pseudopotentials == {"Fe": "Fe_pseudo.UPF"}


def test_dft_config_defaults() -> None:
    """Test default values for optional fields."""
    config = DFTConfig(
        code="quantum_espresso",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": "Fe.UPF"},
    )
    assert config.mixing_beta == 0.7
    assert config.smearing_type == "mv"
    assert config.smearing_width == 0.1
    assert config.diagonalization == "david"


def test_dft_config_invalid_mixing_beta() -> None:
    """Test invalid mixing_beta (must be 0 < beta <= 1)."""
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            mixing_beta=1.5,
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            mixing_beta=-0.1,
        )


def test_dft_config_invalid_smearing_width() -> None:
    """Test invalid smearing_width (must be > 0)."""
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            smearing_width=-0.1,
        )


def test_dft_config_extra_forbid() -> None:
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            extra_field="invalid",  # type: ignore
        )
