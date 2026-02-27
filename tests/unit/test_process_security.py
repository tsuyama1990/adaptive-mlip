import os
from pathlib import Path

import pytest

from pyacemaker.utils.process import run_command


def test_run_command_whitelist() -> None:
    # Command not in whitelist should fail
    with pytest.raises(ValueError, match="is not in the allowed list"):
        run_command(["echo", "hello"])

    # Command in whitelist but with dangerous chars should fail
    with pytest.raises(ValueError, match="contains potentially dangerous character"):
        run_command(["pace_yaml2yace", "config.yaml; rm -rf /"])

@pytest.mark.skipif(os.name == 'nt', reason="Testing bash script mock on Unix")
def test_run_command_success(tmp_path: Path) -> None:
    # Mocking a whitelist binary to just test the runner
    # We would need a real binary in PATH or mock subprocess
    with pytest.MonkeyPatch.context() as m:
        # Instead of actually running, mock subprocess.run to return success
        import pyacemaker.utils.process

        class MockProc:
            returncode = 0
            stdout = "success"
            stderr = ""

        m.setattr(pyacemaker.utils.process.subprocess, "run", lambda *args, **kwargs: MockProc())

        res = run_command(["pace_train", "--config", "x.yaml"], cwd=str(tmp_path))
        assert res.returncode == 0
