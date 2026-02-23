import pytest
from ase import Atoms

from pyacemaker.constants import TEST_ENERGY_GENERIC
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator


# Mock the QEDriver to avoid running real QE
# For UAT, we want to test the flow, but we can't run QE.
# We will monkeypatch the driver used by DFTManager.
class MockQEDriver:
    def __init__(self, should_fail_scf: bool = False, should_fail_setup: bool = False) -> None:
        self.should_fail_scf = should_fail_scf
        self.should_fail_setup = should_fail_setup
        self.call_count = 0

    def get_calculator(self, atoms: Atoms, config: DFTConfig) -> MockCalculator:
        self.call_count += 1

        fail_count = 0
        if self.should_fail_scf:
            fail_count = 1 if self.call_count == 1 else 0

        return MockCalculator(fail_count=fail_count, setup_error=self.should_fail_setup)


@pytest.fixture
def uat_dft_config(tmp_path) -> DFTConfig:
    (tmp_path / "H.UPF").touch()
    (tmp_path / "O.UPF").touch()

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": str(tmp_path / "H.UPF"), "O": str(tmp_path / "O.UPF")},
    )


def test_uat_02_01_single_point_calculation(uat_dft_config: DFTConfig) -> None:
    """
    Scenario 02-01: Calculate Energy & Forces
    Verify that the system can compute properties for a simple structure.
    """
    # Preparation
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True)

    # Inject Mock Driver
    mock_driver = MockQEDriver()
    manager = DFTManager(uat_dft_config, driver=mock_driver)  # type: ignore[arg-type]

    # Action
    # Consume generator
    results = list(manager.compute(iter([atoms])))

    # Expectation
    assert len(results) == 1
    res_atoms = results[0]

    # Check energy (MockCalculator returns generic constant)
    assert res_atoms.get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]

    # Check forces shape (3 atoms, 3 dims)
    forces = res_atoms.get_forces()  # type: ignore[no-untyped-call]
    assert forces.shape == (3, 3)


def test_uat_02_02_self_healing(uat_dft_config: DFTConfig) -> None:
    """
    Scenario 02-02: Self-Healing on SCF Convergence Failure
    Verify that the system retries with adjusted parameters when SCF fails.
    """
    # Preparation
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Inject Mock Driver configured to fail once
    mock_driver = MockQEDriver(should_fail_scf=True)
    manager = DFTManager(uat_dft_config, driver=mock_driver)  # type: ignore[arg-type]

    # Action
    results = list(manager.compute(iter([atoms])))

    # Expectation
    assert len(results) == 1
    assert results[0].get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]

    # Verify it retried (call count > 1)
    assert mock_driver.call_count > 1
