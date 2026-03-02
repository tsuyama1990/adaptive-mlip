from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import DFTConfig
from tests.conftest import create_dummy_pseudopotentials


def test_dft_config_full_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full initialization of DFTConfig with all optional fields."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe_pseudo"])

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


def test_dft_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test default values for optional fields."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

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


@pytest.mark.parametrize("beta", [1.5, -0.1, 0.0])
def test_dft_config_invalid_mixing_beta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beta: float
) -> None:
    """Test invalid mixing_beta (must be 0 < beta <= 1)."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            mixing_beta=beta,
        )


@pytest.mark.parametrize("width", [-0.1, 0.0])
def test_dft_config_invalid_smearing_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: float
) -> None:
    """Test invalid smearing_width (must be > 0)."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            smearing_width=width,
        )


def test_dft_config_extra_forbid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that extra fields are forbidden."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

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


def test_dft_config_external_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that external paths (absolute or relative) are allowed if file exists."""
    monkeypatch.chdir(tmp_path)

    # Create a file in parent directory (simulates system library)
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.UPF"
    outside_file.touch()

    try:
        # Case 1: Absolute path to outside file -> Allowed
        config = DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": str(outside_file.absolute())},
        )
        assert config.pseudopotentials["Fe"] == str(outside_file.absolute())

        # Case 2: Relative path to outside file -> Allowed
        # Construct relative path from tmp_path to outside_file
        rel_path = "../outside_dir/secret.UPF"
        config = DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": rel_path},
        )
        assert config.pseudopotentials["Fe"] == rel_path

    finally:
        # Cleanup
        if outside_file.exists():
            outside_file.unlink()
        if outside_dir.exists():
            outside_dir.rmdir()


def test_dft_config_symlinks_forbidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that symlinks are forbidden."""
    monkeypatch.chdir(tmp_path)

    real_file = tmp_path / "real.UPF"
    real_file.touch()
    symlink_file = tmp_path / "link.UPF"
    symlink_file.symlink_to(real_file)

    with pytest.raises(ValidationError, match="Symlinks are not allowed"):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "link.UPF"},
        )


def test_dft_config_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_dft_config_embedding_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validation of embedding_buffer."""
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

    # Valid buffer
    config = DFTConfig(
        code="qe",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": "Fe.UPF"},
        embedding_buffer=10.0,
    )
    assert config.embedding_buffer == 10.0

    # Invalid buffer (<= 0)
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            embedding_buffer=0.0,
        )

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            embedding_buffer=-5.0,
        )
