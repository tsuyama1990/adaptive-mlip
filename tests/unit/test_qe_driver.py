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
def mock_dft_config(dummy_pseudopotentials_dir: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(dummy_pseudopotentials_dir)
    create_dummy_pseudopotentials(dummy_pseudopotentials_dir, ["H"])

    return DFTConfig(
        code="qe",
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

    # We do not need to patch Espresso or get_calculator. We just want to check
    # that the driver generates the correct k-points via its internal _calculate_kpoints_cached method
    # Since QEDriver._calculate_kpoints_cached is what actually does the math,
    # we can test that directly without instantiating Espresso.
    cell = atoms.get_cell()  # type: ignore[no-untyped-call]
    cell_tuple = tuple(tuple(float(x) for x in row) for row in cell)
    pbc_tuple = tuple(atoms.get_pbc())  # type: ignore[no-untyped-call]

    kpts = driver._calculate_kpoints_cached(cell_tuple, pbc_tuple, mock_dft_config.kpoints_density)

    # Calculate expected k
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

    cell = atoms.get_cell()  # type: ignore[no-untyped-call]
    cell_tuple = tuple(tuple(float(x) for x in row) for row in cell)
    pbc_tuple = tuple(atoms.get_pbc())  # type: ignore[no-untyped-call]

    kpts = driver._calculate_kpoints_cached(cell_tuple, pbc_tuple, mock_dft_config.kpoints_density)

    # Zero length -> treated as non-periodic direction or just handled safely
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

    from pydantic import ValidationError

    # Empty pseudopotential dict - should fail validation at DFTConfig level
    with pytest.raises(ValidationError):
        driver.get_calculator(
            atoms,
            DFTConfig(
                pseudopotentials={},
                code="qe",
                encut=100.0,
                kpoints_density=0.04,
                functional="PBE",
            ),
        )


def test_qe_driver_parameters(mock_dft_config: DFTConfig) -> None:
    """Test that parameters from config are passed to Espresso wrapper correctly without initializing real Espresso."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    # We test the parameters passed to Espresso by mocking it,
    # but we mock it correctly to avoid instantiation issues.
    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kwargs = MockEspresso.call_args.kwargs
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


def test_qe_driver_directory_argument(
    mock_dft_config: DFTConfig, dummy_pseudopotentials_dir: Path
) -> None:
    """Test that directory argument is passed to Espresso."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()
    test_dir = str(dummy_pseudopotentials_dir / "test_dir")

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config, directory=test_dir)
        kwargs = MockEspresso.call_args.kwargs
        assert kwargs.get("directory") == test_dir

    # Default case
    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kwargs = MockEspresso.call_args.kwargs
        assert kwargs.get("directory") == "."
