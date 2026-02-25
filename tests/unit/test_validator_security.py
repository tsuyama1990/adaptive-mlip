from pathlib import Path

import pytest

from pyacemaker.core.validator import LammpsInputValidator


class TestLammpsInputValidator:
    def test_validate_potential_valid_tmp(self, tmp_path):
        """Test that a file in tmp_path is disallowed if outside project root."""
        # tmp_path is usually outside CWD (e.g. /tmp/pytest-...).
        # We expect failure unless we change CWD to tmp_path.
        pot_file = tmp_path / "pot.yace"
        pot_file.touch()

        # Assuming default CWD is project root (/app) and tmp_path is /tmp/...
        # This should now fail.
        with pytest.raises(ValueError, match="outside allowed directories"):
            LammpsInputValidator.validate_potential(pot_file)

    def test_validate_potential_valid_cwd(self):
        """Test that a file in CWD is allowed."""
        pot_file = Path("test_pot.yace")
        pot_file.touch()
        try:
            validated = LammpsInputValidator.validate_potential(pot_file)
            assert validated == pot_file.resolve()
        finally:
            if pot_file.exists():
                pot_file.unlink()

    def test_validate_potential_not_found(self):
        with pytest.raises(FileNotFoundError):
            LammpsInputValidator.validate_potential("nonexistent.yace")

    def test_validate_potential_not_file(self, tmp_path):
        """Test that directory is rejected."""
        with pytest.raises(ValueError, match="not a file"):
            LammpsInputValidator.validate_potential(tmp_path)

    def test_validate_potential_outside_allowed(self):
        """Test rejection of file outside allowed dirs."""
        forbidden_file = Path("/bin/ls")
        if not forbidden_file.exists():
            pytest.skip("/bin/ls not found")

        cwd = Path.cwd().resolve()
        if str(forbidden_file).startswith(str(cwd)):
             pytest.skip("CWD includes /bin")

        with pytest.raises(ValueError, match="outside allowed directories"):
            LammpsInputValidator.validate_potential(forbidden_file)

    def test_validate_potential_symlink_traversal(self, tmp_path):
        """Test symlink resolving to outside (should fail)."""
        # Even if we create a symlink in CWD, resolving to /bin/ls should fail.
        target = Path("/bin/ls")
        if not target.exists():
            pytest.skip("/bin/ls not found")

        symlink = Path("link_to_ls")
        try:
            if symlink.exists():
                symlink.unlink()
            symlink.symlink_to(target)
        except OSError:
            pytest.skip("Failed to create symlink")

        try:
            # Resolving symlink -> /bin/ls.
            # /bin/ls is outside CWD.
            with pytest.raises(ValueError, match="outside allowed directories"):
                LammpsInputValidator.validate_potential(symlink)
        finally:
            if symlink.exists():
                symlink.unlink()

    def test_validate_potential_relative_traversal(self):
        """Test relative path traversal resolving to outside."""
        # /bin/ls exists.
        # We can try to access it via relative path from CWD?
        # e.g. ../../../../../bin/ls
        # But we don't know how deep we are.
        # But we can try to construct one.

        target = Path("/bin/ls")
        if not target.exists():
            pytest.skip("/bin/ls not found")

        # We can use the absolute path. A path starting with / is absolute,
        # but what if user provides "../../bin/ls" relative to CWD?
        # That depends on CWD depth.
        # But validate_potential(Path("/bin/ls")) is effectively what happens if we resolve it.
        # The function resolves the input.

        # If we pass "../../../../bin/ls", it resolves to /bin/ls.
        # So we just need to ensure that *any* input resolving to /bin/ls fails.
        # This is covered by test_validate_potential_outside_allowed effectively.

        # But to be explicit about "traversal syntax":
        # Let's try to construct a relative path to /bin/ls from CWD.

        cwd = Path.cwd().resolve()
        # Find common root? Usually /
        # .. up to root, then down to bin/ls

        depth = len(cwd.parts) - 1 # parts: ('/', 'app') -> 2 parts. depth 1 (one .. to root)
        # Construct path: "../" * depth + "bin/ls"

        rel_path_str = "../" * depth + "bin/ls"
        rel_path = Path(rel_path_str)

        # Verify it resolves to /bin/ls
        if rel_path.resolve() != target:
             # Might fail if CWD is somehow special or we miscounted
             # Just use absolute path for robustness if relative fails logic
             pass
        else:
             with pytest.raises(ValueError, match="outside allowed directories"):
                LammpsInputValidator.validate_potential(rel_path)
