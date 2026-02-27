from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.validator import LammpsInputValidator
from pyacemaker.utils.path import validate_path_safe


class TestLammpsInputValidator:
    def test_validate_potential_symlink_internal(self, tmp_path):
        """Test allowing internal symlink if it resolves safely?
        Requirement: "validate_path_safe function allows paths in /dev/shm but doesn't verify if the directory exists"
        Audit Feedback: "Security: Path Validation Bypass... Add symlink resolution check and canonical path verification."
        My implementation bans symlinks entirely.
        """
        # Create a valid file
        real_file = tmp_path / "real.yace"
        real_file.touch()

        # Create a symlink to it in the same safe dir
        link_file = tmp_path / "link.yace"
        link_file.symlink_to(real_file)

        # This should fail now because we strictly ban symlinks
        with pytest.raises(ValueError, match="Symlinks are not allowed"):
            LammpsInputValidator.validate_potential(link_file)

    def test_validate_potential_symlink_traversal(self, tmp_path):
        """Test detection of symlinks pointing outside."""
        # Create a dummy file outside allowed (simulated by creating a new temp dir which is allowed,
        # so we need to symlink to something NOT allowed, like /etc/passwd or /root if accessible)

        # Actually, `validate_path_safe` checks root allowed list.
        # If we symlink to /etc/passwd, it resolves to /etc/passwd.
        # /etc/passwd is not in allowed roots (cwd, tmp).

        # We can simulate this by mocking allowed roots or resolving.
        # But we added `is_symlink()` check which fails faster.

        link_file = tmp_path / "evil_link.yace"
        # Point to something existing but restricted?
        # Or just any symlink.
        # Since we ban symlinks, any symlink should fail.

        # We don't need a real target if we check is_symlink first?
        # Python `symlink_to` creates it.
        link_file.symlink_to("/etc/passwd")

        with pytest.raises(ValueError, match="Symlinks are not allowed"):
            LammpsInputValidator.validate_potential(link_file)
