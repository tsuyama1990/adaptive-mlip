import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyacemaker.domain_models.constants import (
    EON_CMD,
    EON_CONFIG_FILE,
    EON_DRIVER_SCRIPT,
    EON_FORMAT,
    EON_RESULT_FILE,
)
from pyacemaker.domain_models.eon import EONConfig

logger = logging.getLogger(__name__)


class EONWrapper:
    """
    Wrapper for EON (Adaptive Kinetic Monte Carlo).
    """

    def __init__(self, config: EONConfig) -> None:
        self.config = config

    def run(
        self,
        potential_path: Path,
        work_dir: Path,
        elements: list[str] | None = None,
    ) -> None:
        """
        Sets up and runs an EON simulation.

        Args:
            potential_path: Path to the ACE potential file.
            work_dir: Directory to run the simulation in.
            elements: List of chemical symbols (e.g., ["Fe", "Pt"]) corresponding to EON types.
                      If None, assumes the potential can handle type 1, 2... or logic handles it.
                      Required if using LAMMPS/ACE which needs symbols.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate Driver Script (Buffered Write)
        driver_path = self._generate_pace_driver(work_dir, potential_path, elements)

        # Security: Validate driver path is within work_dir to prevent arbitrary chmod
        try:
            driver_path.resolve().relative_to(work_dir.resolve())
        except ValueError as e:
            msg = f"Driver path {driver_path} is outside work directory {work_dir}"
            raise ValueError(msg) from e

        # Make executable
        driver_path.chmod(0o755)

        # 2. Generate config.ini
        self._generate_config(work_dir, driver_path)

        # 3. Run EON Client
        logger.info(f"Starting EON simulation in {work_dir}")

        eon_executable = shutil.which(EON_CMD)
        if not eon_executable:
            logger.warning(f"{EON_CMD} executable not found. Skipping execution.")
            return

        try:
            # We assume initial structure (reactant.con) is already placed by the scenario.
            # Security: Use list for subprocess args, no shell=True
            result = subprocess.run(
                [eon_executable],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("EON simulation completed.")
            if result.stdout:
                logger.debug(f"EON stdout: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.exception("EON execution failed")
            msg = f"EON execution failed: {e}"
            raise RuntimeError(msg) from e

    def _generate_config(self, work_dir: Path, driver_path: Path) -> None:
        """Generates config.ini for EON."""
        lines = [
            "[Main]",
            f"job = {self.config.job_type}",
            f"temperature = {self.config.temperature}",
            f"confidence = {self.config.confidence}",
            f"random_seed = {self.config.random_seed}",
            "",
            "[Potential]",
            f"type = {self.config.potential_type}",
            f"command = {sys.executable} {driver_path.resolve()}",
            "",
            "[Saddle Search]",
            f"method = {self.config.search_method}",
            "",
        ]
        (work_dir / EON_CONFIG_FILE).write_text("\n".join(lines))

    def _generate_pace_driver(
        self, work_dir: Path, potential_path: Path, elements: list[str] | None = None
    ) -> Path:
        """
        Generates the python script that EON calls to get energy/forces.
        Using ASE to handle I/O and LAMMPS for calculation.
        """

        elements_repr = repr(elements) if elements else "None"
        pot_path_str = str(potential_path.resolve())
        eon_fmt = EON_FORMAT

        # Use buffered template or f-string
        # Note: We are generating code. Special care for injection.
        # potential_path is validated by MDConfig.
        # elements list is from internal logic (scenario).

        # We construct the script content line by line or using a safe template
        script_lines = [
            "#!/usr/bin/env python3",
            "import sys",
            "import numpy as np",
            "from ase.io import read",
            "from ase.calculators.lammpslib import LAMMPSlib",
            "",
            "def main():",
            "    try:",
            "        # Configuration injected by PyAceMaker",
            f"        elements = {elements_repr}",
            f"        pot_path = '{pot_path_str}'",
            "",
            "        # Read input from stdin (EON .con format)",
            f"        atoms = read(sys.stdin, format='{eon_fmt}')",
            "",
            "        # Map types to symbols",
            "        if elements:",
            "            # EON uses 1-based type indices in .con files",
            "            # ASE read maps them to H, He... based on Z=type",
            "            numbers = atoms.get_atomic_numbers()",
            "            symbols = []",
            "            for z in numbers:",
            "                idx = z - 1",
            "                if 0 <= idx < len(elements):",
            "                    symbols.append(elements[idx])",
            "                else:",
            "                    symbols.append('X') # Unknown",
            "            atoms.set_chemical_symbols(symbols)",
            "",
            "        # Setup Calculator",
            "        # Construct LAMMPS commands",
            "        if elements:",
            '            elems_str = " ".join(elements)',
            "        else:",
            '            elems_str = " ".join(sorted(list(set(atoms.get_chemical_symbols()))))',
            "",
            "        # We need to ensure we use absolute path for potential in LAMMPS",
            "        # LAMMPSlib requires lammps-python",
            "",
            "        calc = LAMMPSlib(",
            "            lsp_cmd=None,",
            '            lammps_header=["units metal", "atom_style atomic", "boundary p p p"],',
            '            amendments=[f"pair_style pace", f"pair_coeff * * {pot_path} {elems_str}"],',
            "            log_file=None",
            "        )",
            "",
            "        atoms.calc = calc",
            "",
            "        # Compute",
            "        e = atoms.get_potential_energy()",
            "        forces = atoms.get_forces()",
            "",
            "        # Output in EON expected format (Energy + Forces)",
            "        # Format:",
            "        # Line 1: N",
            "        # Line 2: Energy",
            "        # Line 3-5: Box vectors",
            "        # Line 6+: Type X Y Z Fx Fy Fz",
            "",
            "        print(len(atoms))",
            '        print(f"{e:.12f}")',
            "        cell = atoms.get_cell()",
            "        for i in range(3):",
            '            print(f"{cell[i][0]:.12f} {cell[i][1]:.12f} {cell[i][2]:.12f}")',
            "",
            "        pos = atoms.get_positions()",
            "",
            "        # Reverse mapping: symbol -> index (1-based)",
            "        sym_map = {sym: i+1 for i, sym in enumerate(elements)} if elements else {}",
            "",
            "        for i in range(len(atoms)):",
            "            sym = atoms[i].symbol",
            "            type_idx = sym_map.get(sym, 1) # Default to 1 if not found",
            "            p = pos[i]",
            "            frc = forces[i]",
            '            print(f"{type_idx} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f} {frc[0]:.12f} {frc[1]:.12f} {frc[2]:.12f}")',
            "",
            "    except Exception as e:",
            '        sys.stderr.write(f"Driver Error: {e}\\n")',
            "        sys.exit(1)",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]

        path = work_dir / EON_DRIVER_SCRIPT
        path.write_text("\n".join(script_lines))
        return path

    def parse_results(self, work_dir: Path) -> list[dict[str, Any]]:
        """
        Parses `processtable.dat` from EON output.
        """
        table_path = work_dir / EON_RESULT_FILE
        if not table_path.exists():
            return []

        data = []
        try:
            with table_path.open() as f:
                for raw_line in f:
                    line_content = raw_line.strip()
                    if not line_content or line_content.startswith("#"):
                        continue
                    parts = line_content.split()
                    if len(parts) >= 2:
                        entry = {
                            "id": parts[0],
                            "barrier": float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else 0.0,
                        }
                        data.append(entry)
        except Exception as e:
            logger.warning(f"Failed to parse {EON_RESULT_FILE}: {e}")

        return data
