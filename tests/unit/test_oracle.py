
import tempfile

from ase.build import bulk

from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle
from pyacemaker.domain_models.dft import DFTConfig


def test_mace_manager():
    manager = MACEManager()
    structure = bulk("Cu", "fcc", a=3.6)

    # Needs to be an iterator
    result_iter = manager.compute(iter([structure]))
    result = next(result_iter)

    assert "energy" in result.info
    assert "forces" in result.arrays
    assert "mace_uncertainty" in result.arrays

def test_tiered_oracle_low_uncertainty():
    mace = MACEManager()

    # Create a dummy UPF file to pass validation
    with tempfile.NamedTemporaryFile(suffix=".UPF", delete=False) as f:
        f.write(b"<UPF\nMock File\n")
        dummy_upf = f.name

    try:
        # Create dummy DFT config and manager
        dft_config = DFTConfig(
            code="qe", functional="pbe", kpoints_density=0.1, encut=400,
            pseudopotentials={"Cu": dummy_upf}
        )
        dft = DFTManager(config=dft_config)

        # Set high threshold so MACE is always used
        oracle = TieredOracle(mace_manager=mace, dft_manager=dft, threshold=100.0)

        structure = bulk("Cu", "fcc", a=3.6)
        result_iter = oracle.compute(iter([structure]))
        result = next(result_iter)

        assert "mace_uncertainty" in result.arrays
    finally:
        from pathlib import Path
        Path(dummy_upf).unlink()
