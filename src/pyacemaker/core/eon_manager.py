import subprocess
import sys
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from pyacemaker.interfaces.process import SubprocessRunner


class EONManager:
    """
    Manages EON (Adaptive Kinetic Monte Carlo) simulations.
    Orchestrates configuration generation, driver creation, and execution via EONWrapper.
    """

    def __init__(self, config: EONConfig) -> None:
        self.config = config
        self.wrapper = EONWrapper(config, runner=SubprocessRunner())

    def _write_driver(self, work_dir: Path, potential_path: Path) -> Path:
        """Writes the python driver script that EON will execute."""
        driver_path = work_dir / "pace_driver.py"

        script_content = f"""#!{sys.executable}
import sys
from pyacemaker.interfaces.eon_driver import run_driver

# Configuration
POTENTIAL_PATH = "{potential_path.absolute()}"
THRESHOLD = {self.config.otf_threshold}

if __name__ == "__main__":
    sys.exit(run_driver(POTENTIAL_PATH, THRESHOLD))
"""
        driver_path.write_text(script_content)
        driver_path.chmod(0o755)
        return driver_path

    def run(self, work_dir: Path, potential_path: Path, structure: Atoms) -> dict[str, Any]:
        """
        Runs the EON simulation.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write initial structure
        write(work_dir / "reactant.con", structure, format="eon")

        # Write driver script
        self._write_driver(work_dir, potential_path)

        # Generate config using wrapper logic
        # But wrapper.generate_config uses relative path './pace_driver.py' hardcoded.
        # This matches what we write.
        self.wrapper.generate_config(work_dir / "config.ini")

        try:
            # We use wrapper.run() but we need to handle return codes specifically for OTF halt.
            # EONWrapper.run() raises RuntimeError wrapping CalledProcessError if check=True and ret != 0.

            self.wrapper.run(work_dir)

        except RuntimeError as e:
            if isinstance(e.__cause__, subprocess.CalledProcessError):
                cpe = e.__cause__
                if cpe.returncode == 100:
                    return {
                        "halted": True,
                        "halt_structure_path": str(work_dir / "bad_structure.cfg")
                    }
                # Reraise original RuntimeError (which wraps CPE)
                raise
            # If it's another RuntimeError, raise it
            raise
        except FileNotFoundError as e:
             msg = f"EON executable not found: {self.config.eon_executable}"
             raise RuntimeError(msg) from e

        return {"halted": False}
