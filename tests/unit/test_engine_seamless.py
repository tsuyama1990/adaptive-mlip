from pathlib import Path

from typing import Any
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_engine_resume(tmp_path: Path, mocker: Any) -> None:
    config = MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=1000)
    engine = LammpsEngine(config)

    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Mock file generation
    ctx = mocker.MagicMock()
    ctx.name = tmp_path

    mocker.patch.object(engine.file_manager, "prepare_workspace", return_value=(ctx, tmp_path/"data", tmp_path/"dump", tmp_path/"log", ["H"]))
    mocker.patch.object(engine.generator, "write_script")
    mock_driver = mocker.patch("pyacemaker.core.engine.LammpsDriver").return_value
    mock_driver.extract_variable.side_effect = lambda x: 100.0 if x != "step" else 500
    mock_driver.get_forces.return_value = mocker.MagicMock(tolist=lambda: [[0.0, 0.0, 0.0]])
    mock_driver.get_stress.return_value = mocker.MagicMock(tolist=lambda: [0.0] * 6)

    pot_file = tmp_path / "mock.yace"
    pot_file.touch()

    engine.run(atoms, potential=str(pot_file), resume_from_step=500)

    # Check that resume_from_step was passed to generator
    engine.generator.write_script.assert_called_once()
    kwargs = engine.generator.write_script.call_args[1]
    assert kwargs.get("resume_from_step") == 500
