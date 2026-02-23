from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.constants import RECIPROCAL_FACTOR
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver


@pytest.fixture
def mock_dft_config() -> DFTConfig:
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
        ([False, False, False], 0.0), # Factor 0.0 implies result is 1 (max(1, 0))
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
        k_val = int(np.ceil((RECIPROCAL_FACTOR / 0.04) / 10.0))

        expected_kpts = []
        for _i, is_pbc in enumerate(pbc):
            if is_pbc:
                expected_kpts.append(k_val)
            else:
                expected_kpts.append(1)

        assert kpts == tuple(expected_kpts)


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
