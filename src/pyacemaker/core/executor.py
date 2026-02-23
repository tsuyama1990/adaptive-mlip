import shutil
import subprocess
from pathlib import Path

from pyacemaker.core.exceptions import TrainerError
from pyacemaker.utils.process import run_command


class PacemakerExecutor:
    """
    Handles execution of Pacemaker commands.
    Extracted from PacemakerTrainer for Single Responsibility Principle.
    """

    def run_training(self, input_yaml_path: Path) -> None:
        """
        Executes the pace_train command.

        Args:
            input_yaml_path: Path to the input configuration file.

        Raises:
            TrainerError: If execution fails or executable is missing.
        """
        executable = "pace_train"
        if not shutil.which(executable):
            msg = f"Executable '{executable}' not found in PATH."
            raise TrainerError(msg)

        cmd = [executable, str(input_yaml_path)]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"Training failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            msg = f"Training failed unexpectedly: {e}"
            raise TrainerError(msg) from e
