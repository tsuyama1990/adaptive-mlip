from ase import Atoms

from pyacemaker.core.mock_oracle import MockOracle


def test_mock_oracle_compute_success():
    """Test MockOracle computes energies and forces."""
    oracle = MockOracle()
    atoms = Atoms("He", positions=[[0, 0, 0]])

    # Process
    results = list(oracle.compute(iter([atoms])))
    assert len(results) == 1

    res_atoms = results[0]
    assert "energy" in res_atoms.info
    assert "forces" in res_atoms.arrays
    assert res_atoms.info["provenance"] == "MOCK_ORACLE"

def test_mock_oracle_compute_failure(caplog):
    """Test MockOracle handles calculation failures gracefully (skips atom and logs error)."""
    import logging
    oracle = MockOracle()
    atoms = Atoms("He", positions=[[0, 0, 0]])

    # Mock LennardJones to raise ValueError
    from unittest.mock import patch

    # We patch ase.calculators.lj.LennardJones.get_potential_energy to raise
    with patch("ase.calculators.lj.LennardJones.get_potential_energy", side_effect=ValueError("Test Failure")):
        with caplog.at_level(logging.ERROR):
            # Use simple iteration to verify 0 yields
            count = 0
            for _ in oracle.compute(iter([atoms])):
                count += 1
            # Should be skipped
            assert count == 0

            # Verify error was logged
            assert "MockOracle computation failed" in caplog.text
            assert "Test Failure" in caplog.text
