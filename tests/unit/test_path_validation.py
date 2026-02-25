from pathlib import Path

import pytest

from pyacemaker.utils.path import validate_path_safe


def test_validate_path_cwd_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    # Valid local path
    p = tmp_path / "valid.txt"
    assert validate_path_safe(p) == p.resolve()

    # Valid relative path
    assert validate_path_safe(Path("foo.txt")) == (tmp_path / "foo.txt").resolve()

    # Invalid: Absolute path outside CWD (e.g., /etc/passwd)
    # Using a fake path that looks absolute
    outside_path = Path("/tmp/outside_project.txt")
    if outside_path.exists(): # Only works if it resolves
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_safe(outside_path)

    # Since /tmp is usually outside CWD (unless CWD is /tmp), this should raise.
    # But if CWD is /tmp, we need to pick another path.
    # Assuming CWD is not root.

    # Test strict traversal attack
    with pytest.raises(ValueError, match="Path traversal attempt"):
        validate_path_safe(Path("../secret.txt"))

def test_validate_path_symlink_attack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    # Create symlink pointing to a system file that is definitely outside allowed roots
    # Allowed: CWD (/app), Temp (/tmp/...), /dev/shm
    # Target: /usr/bin/ls (usually safe to assume exists)
    target = Path("/usr/bin/ls")
    if not target.exists():
        pytest.skip("/usr/bin/ls not found")

    symlink = tmp_path / "link_to_system"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks not supported")

    # Validation should fail because resolved path is outside
    with pytest.raises(ValueError, match="(Path traversal detected|Invalid path resolution)"):
        validate_path_safe(symlink)
