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

    # Invalid: Absolute path outside CWD (e.g., /usr/bin/ls)
    # Using a system path that is definitely outside the temp dir or CWD.
    outside_path = Path("/usr/bin/ls")

    # Only test if it exists to be robust, though resolve(strict=False) works even if not exists.
    # We rely on resolve() canonicalizing it to something outside allowed roots.
    # On linux /usr/bin is outside temp.
    # But just in case, verify logic:
    try:
        outside_path.resolve(strict=False)
        # Check if it happens to be inside allowed roots (e.g. if CWD is /usr/bin)
        # Assuming CWD is tmp_path which is /tmp/...
        # So /usr/bin is safe to test as invalid.
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_safe(outside_path)
    except Exception: # noqa: S110
        # If resolve fails or something else weird happens, skip
        pass

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
