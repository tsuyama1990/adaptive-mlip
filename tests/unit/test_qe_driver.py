from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import RECIPROCAL_FACTOR
from pyacemaker.interfaces.qe_driver import QEDriver
from tests.conftest import create_dummy_pseudopotentials


@pytest.fixture
def mock_dft_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["H"])

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.6,
        smearing_type="mv",
        smearing_width=0.02,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF"},
    )


@pytest.mark.parametrize(
    ("pbc", "expected_factor"),
    [
        ([True, True, True], 1.0),
        ([False, False, False], 0.0),  # Factor 0.0 implies result is 1 (max(1, 0))
        ([True, True, False], 1.0),
    ],
)
def test_qe_driver_kpoints_parametrized(
    mock_dft_config: DFTConfig, pbc: list[bool], expected_factor: float
) -> None:
    """Test k-point generation with various PBC settings."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=pbc)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")
        assert kpts is not None

        # Calculate expected k
        # If factor is 1.0 (PBC), use formula. If 0.0 (No PBC), expect 1.
        # Formula N = ceil( (2*pi / spacing) / L )
        # Here spacing=0.04, L=10.0.
        k_val = int(np.ceil((RECIPROCAL_FACTOR / 0.04) / 10.0))

        expected_kpts = []
        for is_pbc in pbc:
            if is_pbc:
                expected_kpts.append(k_val)
            else:
                expected_kpts.append(1)

        assert kpts == tuple(expected_kpts)


def test_qe_driver_kpoints_zero_length(mock_dft_config: DFTConfig) -> None:
    """Test k-point generation with zero-length cells (should default to 1)."""
    # Cell with zero volume or very small dimensions
    atoms = Atoms("H", cell=[0.0, 0.0, 0.0], pbc=True)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")

        # Zero length -> treated as non-periodic direction or just handled safely
        # Implementation uses mask (lengths >= 1e-3). So should be 1.
        assert kpts == (1, 1, 1)


def test_qe_driver_invalid_input(mock_dft_config: DFTConfig) -> None:
    """Test validation of invalid inputs."""
    driver = QEDriver()
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Negative Energy Cutoff
    mock_dft_config.encut = -10.0
    with pytest.raises(ValueError, match="Energy cutoff must be positive"):
        driver.get_calculator(atoms, mock_dft_config)
    mock_dft_config.encut = 500.0  # Reset

    # Negative K-point density
    mock_dft_config.kpoints_density = -0.04
    with pytest.raises(ValueError, match="K-points density must be positive"):
        driver.get_calculator(atoms, mock_dft_config)
    mock_dft_config.kpoints_density = 0.04

    # Invalid Pseudopotential Key
    # Pydantic validation happens at init, but we modified attribute.
    # The driver re-validates.
    mock_dft_config.pseudopotentials = {"InvalidElement": "file.upf"}
    with pytest.raises(ValueError, match="Invalid chemical symbol"):
        driver.get_calculator(atoms, mock_dft_config)


def test_qe_driver_parameters(mock_dft_config: DFTConfig) -> None:
    """Test that parameters from config are passed to Espresso."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)

        kwargs = MockEspresso.call_args[1]
        input_data = kwargs.get("input_data", {})

        # Comprehensive check of all parameters
        control = input_data.get("control", {})
        system = input_data.get("system", {})
        electrons = input_data.get("electrons", {})

        assert control["calculation"] == "scf"
        assert control["restart_mode"] == "from_scratch"
        assert control["disk_io"] == "low"

        assert system["ecutwfc"] == 500.0
        assert system["occupations"] == "smearing"
        assert system["smearing"] == "mv"
        assert system["degauss"] == 0.02

        assert electrons["mixing_beta"] == 0.6
        assert electrons["diagonalization"] == "david"
        assert electrons["conv_thr"] == 1.0e-8

        # Check pseudopotentials
        pseudos = kwargs.get("pseudopotentials")
        assert pseudos == {"H": "H.UPF"}
