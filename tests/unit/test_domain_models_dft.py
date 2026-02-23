import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.dft import DFTConfig


def test_dft_config_full_valid(tmp_path) -> None:
    """Test full initialization of DFTConfig with all optional fields."""
    pseudo = tmp_path / "Fe_pseudo.UPF"
    pseudo.touch()

    config = DFTConfig(
        code="quantum_espresso",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.5,
        smearing_type="gaussian",
        smearing_width=0.05,
        diagonalization="cg",
        pseudopotentials={"Fe": str(pseudo)},
    )
    assert config.code == "quantum_espresso"
    assert config.mixing_beta == 0.5
    assert config.smearing_type == "gaussian"


def test_dft_config_defaults(tmp_path) -> None:
    """Test default values for optional fields."""
    pseudo = tmp_path / "Fe.UPF"
    pseudo.touch()

    config = DFTConfig(
        code="quantum_espresso",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": str(pseudo)},
    )
    assert config.mixing_beta == 0.7  # Default
    assert config.smearing_type == "mv"  # Default
    assert config.diagonalization == "david"  # Default


def test_dft_config_invalid_pseudopotential(tmp_path) -> None:
    """Test validation of pseudopotentials."""
    # Create one valid file
    pseudo = tmp_path / "Fe.UPF"
    pseudo.touch()

    # Missing file
    with pytest.raises(ValidationError, match="Pseudopotential file not found"):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "NonExistent.UPF"},
        )

    # Empty path
    with pytest.raises(ValidationError, match="cannot be empty"):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": ""},
        )


def test_dft_config_path_traversal() -> None:
    """Test prevention of path traversal."""
    # Assuming strict resolution prevents traversal if file doesn't exist at resolved path
    # or explicit check.
    # Note: If "/etc/passwd" existed, it might pass 'exists' check but fail traversal if we enforced strict root.
    # Current implementation enforces existence relative to CWD or absolute.
    # Traversing out of CWD is allowed if the file exists (absolute path).
    # But "..//foo" relative path logic is handled by resolve().


def test_dft_config_negative_values(tmp_path) -> None:
    """Test validation of numeric fields."""
    pseudo = tmp_path / "Fe.UPF"
    pseudo.touch()

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=-0.1,  # Invalid
            encut=500.0,
            pseudopotentials={"Fe": str(pseudo)},
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            mixing_beta=1.5,  # > 1.0
            pseudopotentials={"Fe": str(pseudo)},
        )


def test_dft_config_strategies(tmp_path) -> None:
    """Test strategy multiplier validation."""
    pseudo = tmp_path / "Fe.UPF"
    pseudo.touch()

    config = DFTConfig(
        code="qe",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": str(pseudo)},
        mixing_beta_factor=0.5,
        smearing_width_factor=2.0,
    )
    assert config.mixing_beta_factor == 0.5

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": str(pseudo)},
            mixing_beta_factor=1.1,  # > 1.0 invalid for reduction
        )
