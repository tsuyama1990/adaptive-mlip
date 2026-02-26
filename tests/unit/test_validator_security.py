from pathlib import Path

import pytest

from pyacemaker.core.validator import LammpsInputValidator


class TestLammpsInputValidator:
    def test_validate_structure_none(self):
        with pytest.raises(ValueError, match="Structure is None"):
            LammpsInputValidator.validate_structure(None)

    def test_validate_structure_type_error(self):
        with pytest.raises(TypeError, match="Invalid structure type"):
            LammpsInputValidator.validate_structure("not an atoms object")

    def test_validate_structure_empty(self):
        from ase import Atoms
        atoms = Atoms()
        with pytest.raises(ValueError, match="Structure is empty"):
            LammpsInputValidator.validate_structure(atoms)

    def test_validate_structure_zero_volume(self):
        from ase import Atoms
        # Zero volume cell
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[0, 0, 0], pbc=True)
        # Matches error from exception handling block
        with pytest.raises(ValueError, match="Failed to compute structure volume"):
            LammpsInputValidator.validate_structure(atoms)

    def test_validate_potential_none(self):
        with pytest.raises(ValueError, match="Validator requires a potential"):
            LammpsInputValidator.validate_potential(None)

    def test_validate_potential_not_found(self):
        with pytest.raises(FileNotFoundError, match="Potential file not found"):
            LammpsInputValidator.validate_potential("nonexistent.yace")

    def test_validate_potential_outside_allowed(self):
        """Test rejection of file outside allowed dirs."""
        forbidden_file = Path("/bin/ls")
        if not forbidden_file.exists():
            pytest.skip("/bin/ls not found")

        cwd = Path.cwd().resolve()
        if str(forbidden_file).startswith(str(cwd)):
             pytest.skip("CWD includes /bin")

        with pytest.raises(ValueError, match="Potential path is outside allowed directory"):
            LammpsInputValidator.validate_potential(forbidden_file)

    def test_validate_potential_symlink_traversal(self, tmp_path):
        """Test symlink resolving to outside (should fail)."""
        # Create a symlink in tmp_path (allowed location) pointing to /bin/ls (forbidden location)
        target = Path("/bin/ls")
        if not target.exists():
            pytest.skip("/bin/ls not found")

        symlink = tmp_path / "link_to_ls"
        try:
            symlink.symlink_to(target)
        except OSError:
            pytest.skip("Failed to create symlink")

        # Resolving symlink -> /bin/ls.
        # /bin/ls is not in /tmp, not in CWD.
        # So it should raise ValueError.

        with pytest.raises(ValueError, match="Potential path is outside allowed directory"):
            LammpsInputValidator.validate_potential(symlink)
