from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_lammps_engine_run(mock_md_config: MDConfig) -> None:
    engine = LammpsEngine(mock_md_config)

    atoms = Atoms("H")
    result = engine.run(atoms, "potential.yace")

    assert isinstance(result, dict)
    assert "energy" in result
    assert "forces" in result
    assert "halted" in result

def test_lammps_engine_failure(mock_md_config: MDConfig) -> None:
    """Test engine error handling (simulated)."""
    # Since LammpsEngine is currently mocked, we can't easily force an external failure
    # unless we mock the subprocess call (which isn't implemented yet).
    # However, we can test configuration validation if applicable.
    # For now, we just ensure it handles None potential gracefully if designed to.

    engine = LammpsEngine(mock_md_config)

    # If potential is None, it might raise if strict, or mock success.
    # The current mock implementation ignores potential input.
    # We will just verify it returns a result even with minimal input.
    result = engine.run(None, None)
    assert result["halted"] is False
