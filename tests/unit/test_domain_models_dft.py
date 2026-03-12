from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import DFTConfig
from tests.conftest import create_dummy_pseudopotentials


def test_dft_config_full_valid(
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test full initialization of DFTConfig with all optional fields."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe_pseudo"])

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


def test_dft_config_defaults(
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test default values for optional fields."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe"])

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
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch, beta: float
) -> None:
    """Test invalid mixing_beta (must be 0 < beta <= 1)."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe"])

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
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch, width: float
) -> None:
    """Test invalid smearing_width (must be > 0)."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe"])

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            smearing_width=width,
        )


def test_dft_config_extra_forbid(
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that extra fields are forbidden."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe"])

    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.UPF"},
            extra_field="invalid",
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


def test_dft_config_external_paths(
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that external paths (absolute or relative) are strictly denied."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)

    # Case 1: Absolute path -> Denied
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "/non/existent/path/for/sure/fe.upf"},
        )

    # Case 2: Relative path with directory traversal -> Denied
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"Fe": "../outside_dir/secret.UPF"},
        )


def test_dft_config_embedding_buffer(
    dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test validation of embedding_buffer."""
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["Fe"])

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
