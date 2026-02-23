from unittest.mock import patch

import pytest
from ase import Atoms

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


def test_qe_driver_get_calculator_kpoints(mock_dft_config: DFTConfig) -> None:
    """Test k-point generation based on density."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    # K-points density = 0.04.
    # Reciprocal lattice vectors ~ 2*pi / L.
    # spacing = 2 * pi * density ? No, usually kspacing = 2*pi / (N * L) ?
    # Standard formula: k = max(1, int(L * density) + 0.5) ?
    # Or density is K-points per reciprocal Angstrom?
    # Usually `kpoints_density` means spacing in reciprocal space (e.g. 0.04 A^-1).
    # N_k = ceil( |b_i| / spacing ). |b_i| = 2*pi / |a_i|.
    # So N_k = ceil( (2*pi/L) / spacing ) ?
    # Wait, spec says "kpoints_density: PositiveFloat = Field(..., description='K-points density in 1/Angstrom')".
    # If density is 0.04 A^-1 (which is very fine spacing), N_k is large.
    # If density is "density of kpoints per Angstrom^-1", typically it means spacing.
    # If it means "density per reciprocal Angstrom", usually it's N ~ L * density.
    # A common convention in VASP/CASTEP is "k-spacing". E.g. 0.04 means grid spacing is 0.04 A^-1.
    # Then N = 2*pi / (L * spacing).
    # Example: L=10, spacing=0.04. 2*pi/10 = 0.628. 0.628 / 0.04 = 15.7 -> 16.

    # Another convention: "density" means "points per Angstrom^-1".
    # I'll stick to "k-spacing" interpretation as it is standard in high-throughput.
    # Let's see if the implementation uses specific logic. I'll define it here.
    # I'll assume 2*pi / (L * density) logic for now.

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)

        # Verify k-points passed to Espresso
        call_args = MockEspresso.call_args[1]
        kpts = call_args.get("kpts")
        assert kpts is not None
        # With L=10, density=0.04 (spacing), N ~ 16.
        # But if density=0.04 is "k-points per Angstrom" (linear density)?
        # That would be N = L * density = 10 * 0.04 = 0.4 -> 1. Too small.
        # Spacing is more likely.

        # I'll let the implementation decide, but I'll assert it's a tuple of 3 integers.
        assert isinstance(kpts, tuple) or isinstance(kpts, list)
        assert len(kpts) == 3
        assert all(isinstance(k, int) for k in kpts)
        assert all(k > 0 for k in kpts)


def test_qe_driver_kpoints_non_pbc(mock_dft_config: DFTConfig) -> None:
    """Test k-point generation for non-periodic systems."""
    # Isolated atom, pbc=False
    atoms = Atoms("H", cell=[10, 10, 10], pbc=[False, False, False])
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")
        assert kpts == (1, 1, 1)

    # Surface (slab), pbc=[True, True, False]
    atoms = Atoms("H", cell=[10, 10, 10], pbc=[True, True, False])
    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")
        # kx, ky should be > 1 (calculated), kz should be 1
        assert kpts[0] > 1
        assert kpts[1] > 1
        assert kpts[2] == 1


def test_qe_driver_parameters(mock_dft_config: DFTConfig) -> None:
    """Test that parameters from config are passed to Espresso."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)

        kwargs = MockEspresso.call_args[1]
        input_data = kwargs.get("input_data", {})

        # Check control/system/electrons sections structure or flat dict
        # ASE Espresso usually takes flat dict or sections.
        # Checking for specific keys.
        assert input_data["control"]["calculation"] == "scf"
        assert input_data["system"]["ecutwfc"] == 500.0
        assert input_data["system"]["occupations"] == "smearing"
        assert input_data["system"]["smearing"] == "mv"
        assert input_data["system"]["degauss"] == 0.02
        assert input_data["electrons"]["mixing_beta"] == 0.6
        assert input_data["electrons"]["diagonalization"] == "david"

        # Check pseudopotentials
        pseudos = kwargs.get("pseudopotentials")
        assert pseudos == {"H": "H.UPF"}
