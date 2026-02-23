
import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import DFTConfig


def test_dft_config_full_valid(tmp_path, monkeypatch) -> None:
    """Test full initialization of DFTConfig with all optional fields."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe_pseudo.UPF").touch()

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


def test_dft_config_defaults(tmp_path, monkeypatch) -> None:
    """Test default values for optional fields."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

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


def test_dft_config_invalid_mixing_beta(tmp_path, monkeypatch) -> None:
    """Test invalid mixing_beta (must be 0 < beta <= 1)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

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


def test_dft_config_invalid_smearing_width(tmp_path, monkeypatch) -> None:
    """Test invalid smearing_width (must be > 0)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            smearing_width=-0.1,
        )


def test_dft_config_extra_forbid(tmp_path, monkeypatch) -> None:
    """Test that extra fields are forbidden."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            extra_field="invalid",  # type: ignore
        )


def test_dft_config_empty_pseudopotential() -> None:
    """Test that pseudopotential paths cannot be empty strings."""
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": ""},
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "   "},
        )


def test_dft_config_path_traversal(tmp_path, monkeypatch) -> None:
    """Test that path traversal is blocked."""
    monkeypatch.chdir(tmp_path)

    # Create a file in parent directory (risky but simulates outside access)
    # With strict=True in resolve(), the file MUST exist to be resolved.
    # If we pass "../secret.UPF" and it doesn't exist, resolve() raises FileNotFoundError.
    # If it DOES exist, resolve() returns absolute path.
    # Then is_relative_to(cwd) checks if it's inside.

    # Case 1: File doesn't exist -> FileNotFoundError
    with pytest.raises(ValidationError, match="Pseudopotential file not found"):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "../secret.UPF"},
        )

    # Case 2: File exists but outside -> Path traversal detected
    # We need to simulate a file outside CWD.
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.UPF"
    outside_file.touch()

    try:
        # Pass absolute path to outside file
        with pytest.raises(ValidationError, match="Path traversal detected"):
            DFTConfig(
                code="qe",
                functional="PBE",
                kpoints_density=0.04,
                encut=500.0,
                pseudopotentials={"Fe": str(outside_file)},
            )
    finally:
        # Cleanup
        if outside_file.exists():
            outside_file.unlink()
        if outside_dir.exists():
            outside_dir.rmdir()

def test_dft_config_file_not_found(tmp_path, monkeypatch) -> None:
    """Test that non-existent file raises error."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="Pseudopotential file not found"):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "missing.UPF"},
        )
