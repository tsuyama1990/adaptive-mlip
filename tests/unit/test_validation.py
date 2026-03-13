from pathlib import Path

import pytest

from pyacemaker.utils.validation import validate_lammps_command, validate_lammps_script_file


def test_validate_lammps_command_valid() -> None:
    validate_lammps_command("units metal")
    validate_lammps_command("thermo 100")

def test_validate_lammps_command_invalid_chars() -> None:
    with pytest.raises(ValueError, match="contains explicitly blocked shell metacharacters|contains forbidden characters"):
        validate_lammps_command("units metal ; rm -rf /")

def test_validate_lammps_command_invalid_command() -> None:
    with pytest.raises(ValueError, match="forbidden or unrecognized command|contains forbidden characters"):
        validate_lammps_command("rm -rf /")

def test_validate_lammps_script_file_valid(tmp_path: Path) -> None:
    script = tmp_path / "valid.lmp"
    script.write_text("units metal\nthermo 100\n# comment")
    validate_lammps_script_file(script)

def test_validate_lammps_script_file_invalid(tmp_path: Path) -> None:
    script = tmp_path / "invalid.lmp"
    script.write_text("units metal\nrm -rf /")
    with pytest.raises(ValueError, match="Forbidden command detected"):
        validate_lammps_script_file(script)
