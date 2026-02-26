import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from pyacemaker.interfaces.process import ProcessRunner


class MockProcessRunner(ProcessRunner):
    def __init__(self, return_code=0, stdout="", stderr="") -> None:
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
        if self.return_code != 0 and kwargs.get("check"):
            import subprocess
            raise subprocess.CalledProcessError(self.return_code, cmd, self.stdout, self.stderr)
        return mock_process


@pytest.fixture
def mock_potential_path():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        yield Path(tmp.name)


def test_eon_generate_config(mock_potential_path):
    config = EONConfig(
        potential_path=mock_potential_path,
        temperature=500.0,
        random_seed=12345,
        eon_executable="eonclient"
    )
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "config.ini"
        wrapper.generate_config(output_path)

        content = output_path.read_text()
        assert "[Main]" in content
        assert "job = akmc" in content
        assert "temperature = 500.0" in content
        assert "random_seed = 12345" in content
        assert "[Potential]" in content
        assert "potential = command_line" in content
        # Check command path
        assert f"command = {Path('pace_driver.py')!s}" or "python pace_driver.py" in content
        assert "[Structure]" in content
        assert "supercell = [1, 1, 1]" in content
        assert "[Communicator]" in content
        assert "type = local" in content
        assert "client_path = eonclient" in content

        # Verify file permissions (0o600 for config)
        import stat
        st = output_path.stat()
        assert stat.S_IMODE(st.st_mode) == 0o600


def test_eon_generate_driver_script(mock_potential_path):
    config = EONConfig(potential_path=mock_potential_path)
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        driver_path = Path(tmp_dir) / "pace_driver.py"
        wrapper.generate_driver_script(driver_path)

        content = driver_path.read_text()
        assert "from ase" in content or "import ase" in content
        assert "PACE_POTENTIAL_PATH" in content
        # Check if file is executable (0o700)
        import os
        import stat
        assert os.access(driver_path, os.X_OK)
        st = driver_path.stat()
        assert stat.S_IMODE(st.st_mode) == 0o700


def test_eon_run_command(mock_potential_path):
    # Use a safe path for eon_executable to pass security checks
    with tempfile.NamedTemporaryFile(suffix="eonclient") as tmp_exec:
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

        # Verify command splitting
        assert cmd == ["mpirun", "-np", "4", safe_executable]
        assert run_cwd == cwd
        assert "PACE_POTENTIAL_PATH" in kwargs["env"]
        assert kwargs["env"]["PACE_POTENTIAL_PATH"] == str(mock_potential_path)


def test_eon_run_not_found(mock_potential_path):
    # Test error handling for executable not found
    runner = MockProcessRunner(return_code=127, stderr="not found")

    # Use a safe path for eon_executable to pass security checks
    with tempfile.NamedTemporaryFile(suffix="eonclient") as tmp_exec:
        safe_executable = str(Path(tmp_exec.name))

    config = EONConfig(potential_path=mock_potential_path, eon_executable=safe_executable)
    wrapper = EONWrapper(config, runner=runner)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(RuntimeError) as excinfo:
            wrapper.run(Path(tmp_dir))
        assert "EON executable not found" in str(excinfo.value)

def test_eon_file_write_failure(mock_potential_path):
    config = EONConfig(potential_path=mock_potential_path)
    wrapper = EONWrapper(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "config.ini"

        # Mock pathlib.Path.write_text to fail
        with patch.object(Path, "write_text", side_effect=OSError("Permission denied")):
            with pytest.raises(RuntimeError) as excinfo:
                wrapper.generate_config(output_path)
            assert "Failed to write file" in str(excinfo.value)
