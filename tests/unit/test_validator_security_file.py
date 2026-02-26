
import pytest

from pyacemaker.core.validator import LammpsInputValidator


class TestLammpsInputValidatorFile:

    def test_validate_potential_exists(self, tmp_path):
        # Create a valid potential file
        pot_file = tmp_path / "test.yace"
        pot_file.touch()

        # Should not raise
        LammpsInputValidator.validate_potential(str(pot_file))

    def test_validate_potential_not_exists(self, tmp_path):
        # Path to non-existent file
        pot_file = tmp_path / "missing.yace"

        with pytest.raises(FileNotFoundError):
            LammpsInputValidator.validate_potential(str(pot_file))

    def test_validate_potential_unsafe(self, tmp_path):
        # Unsafe characters
        pot_file = tmp_path / "test;echo.yace"

        # Should be caught by validate_path_safe inside validate_potential
        with pytest.raises(ValueError, match="Path contains invalid characters"):
            LammpsInputValidator.validate_potential(str(pot_file))
