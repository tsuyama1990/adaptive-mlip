from pathlib import Path

import pytest

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from tests.unit.mock_process import MockProcessRunner


@pytest.fixture
def valid_eon_config(tmp_path: Path) -> EONConfig:
    pot = tmp_path / "pot.yace"
    pot.touch()
    return EONConfig(potential_path=pot, temperature=500.0)

def test_eon_wrapper_init(valid_eon_config: EONConfig) -> None:
    driver = EONWrapper(valid_eon_config)
    assert driver.config == valid_eon_config


def test_generate_config(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    driver = EONWrapper(valid_eon_config)

    config_path = tmp_path / "config.ini"
    driver.generate_config(config_path)

    assert config_path.exists()
    content = config_path.read_text()
    assert "temperature = 500.0" in content
    assert f"potentials_path = {valid_eon_config.potential_path}" in content


def test_run_success(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    runner = MockProcessRunner()
    driver = EONWrapper(valid_eon_config, runner=runner)

    # Mock existence of required files (validate_path checks paths)
    # validate_path checks if executable path is safe. eonclient is just a string here.
    # If eon_executable was a path, we'd need to mock it.

    driver.run(working_dir=tmp_path)

    assert len(runner.commands) == 1
    cmd, cwd = runner.commands[0]
    assert cmd == ["eonclient"]
    assert cwd == tmp_path


def test_run_failure(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    runner = MockProcessRunner(returncode=1, stderr="Error occurred")
    driver = EONWrapper(valid_eon_config, runner=runner)

    # MockProcessRunner assumes kwargs['check'] = True if returncode != 0
    # But EONWrapper uses runner.run(..., check=True) implicitly or explicitly.
    # The MockProcessRunner logic I wrote in mock_process.py raises CalledProcessError
    # ONLY IF check=True is passed.
    # EONWrapper calls: self.runner.run(cmd, cwd=working_dir)
    # The SubprocessRunner adds defaults. But MockProcessRunner doesn't unless I add them.
    # So EONWrapper calls run(cmd, cwd).
    # MockProcessRunner.run needs to check 'check' kwarg or raise if we want.
    # The real SubprocessRunner sets defaults.
    # Let's fix EONWrapper to explicitly pass check=True to be clear,
    # OR fix MockProcessRunner to default check=True like SubprocessRunner.
    # But wait, EONWrapper just calls runner.run().
    # SubprocessRunner implementation:
    # def run(..., **kwargs):
    #    kwargs.setdefault("check", True) ...
    # MockProcessRunner implementation:
    # def run(..., **kwargs):
    #    if kwargs.get("check", False) ...

    # So if EONWrapper doesn't pass check=True, MockProcessRunner defaults to False.
    # But SubprocessRunner defaults to True.
    # EONWrapper relies on SubprocessRunner's default.
    # Since MockProcessRunner is a test double, it should mimic the behavior we expect *or*
    # we should make EONWrapper explicitly pass check=True to allow strict mocking.

    # Let's update EONWrapper to explicitly pass check=True for clarity and testability.
    # OR update MockProcessRunner.
    # Updating EONWrapper is better design (explicit > implicit).

    # For now, I'll update the test to use a properly configured MockProcessRunner
    # or patch EONWrapper to pass check=True.
    # Actually, EONWrapper logic:
    # result = self.runner.run(cmd, cwd=working_dir)

    # I will modify EONWrapper to pass check=True explicitly in next step.
    # For this test file, I will expect it to fail unless I fix EONWrapper or Mock.
    # I'll fix the expectation here assuming I will fix EONWrapper.

    # Wait, the failure is "Failed: DID NOT RAISE".
    # This means EONWrapper called run(), MockRunner returned mock_res (returncode=1),
    # but NO exception was raised because check=True wasn't passed to MockRunner.
    # EONWrapper didn't check returncode itself because it relies on check=True behavior.

    with pytest.raises(RuntimeError, match="EON execution failed"):
        driver.run(working_dir=tmp_path)


def test_parse_results(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    driver = EONWrapper(valid_eon_config)

    (tmp_path / "dynamics.txt").write_text("Step 1: 0.5 eV barrier\n")
    (tmp_path / "processtable.dat").write_text("Process 1: Barrier 0.5 eV\n")

    results = driver.parse_results(tmp_path)
    assert "dynamics" in results
    assert "processtable" in results
    assert results["dynamics"] == "Step 1: 0.5 eV barrier\n"
