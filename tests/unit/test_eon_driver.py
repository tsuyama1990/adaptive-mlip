import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from pyacemaker.interfaces.process import ProcessRunner


class MockProcessRunner(ProcessRunner):
    def __init__(self, return_code=0, stdout="", stderr=""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.commands = []

    def run(self, cmd, cwd, **kwargs):
        self.commands.append((cmd, cwd, kwargs))
        mock_process = MagicMock()
        mock_process.returncode = self.return_code
        mock_process.stdout = self.stdout
        mock_process.stderr = self.stderr
        return mock_process


@pytest.fixture
def mock_potential_path():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        yield Path(tmp.name)


def test_eon_generate_config(mock_potential_path):
    config = EONConfig(potential_path=mock_potential_path, temperature=500.0)
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "config.ini"
        # We expect generate_config to set potential = command_line and command = python pace_driver.py
        # But wait, generate_config needs to know where the driver script will be or assume it's in CWD.
        wrapper.generate_config(output_path)

        content = output_path.read_text()
        assert "job = akmc" in content
        assert "temperature = 500.0" in content
        assert "potential = command_line" in content
        # The command should point to the driver script. It might use absolute python path.
        assert "pace_driver.py" in content


def test_eon_generate_driver_script(mock_potential_path):
    config = EONConfig(potential_path=mock_potential_path)
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        driver_path = Path(tmp_dir) / "pace_driver.py"
        wrapper.generate_driver_script(driver_path)

        content = driver_path.read_text()
        assert "from ase" in content or "import ase" in content
        # The script uses environment variable for potential path, so it shouldn't be hardcoded in the script content
        # assert str(mock_potential_path) in content
        assert "PACE_POTENTIAL_PATH" in content
        # It should read from stdin (or a file if EON requires)
        # Usually EON command_line potential passes coords via file or stdin.
        # Let's assume stdin/stdout for now as per SPEC.
        assert "sys.stdin" in content or "input()" in content


def test_eon_run_command(mock_potential_path):
    # Use a safe path for eon_executable to pass security checks
    with tempfile.NamedTemporaryFile(suffix="eonclient") as tmp_exec:
        safe_executable = Path(tmp_exec.name).name # Use relative name or resolve to temp
        # Actually validate_path_safe allows temp dir.
        safe_executable = str(Path(tmp_exec.name))

    config = EONConfig(
        potential_path=mock_potential_path,
        mpi_command="mpirun -np 4",
        eon_executable=safe_executable
    )
    runner = MockProcessRunner()
    wrapper = EONWrapper(config, runner=runner)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = Path(tmp_dir)
        wrapper.run(cwd)

        assert len(runner.commands) == 1
        cmd, run_cwd, kwargs = runner.commands[0]

        assert cmd == ["mpirun", "-np", "4", safe_executable]
        assert run_cwd == cwd


def test_eon_parse_results(mock_potential_path):
    config = EONConfig(potential_path=mock_potential_path)
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = Path(tmp_dir)
        (cwd / "dynamics.txt").write_text("step 1 energy -100.0")
        (cwd / "processtable.dat").write_text("process 1 barrier 0.5")

        results = wrapper.parse_results(cwd)
        assert results["dynamics"] == "step 1 energy -100.0"
        assert results["processtable"] == "process 1 barrier 0.5"
