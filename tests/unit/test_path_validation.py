from pathlib import Path

import pytest

from pyacemaker.utils.path import validate_path_safe


def test_validate_path_safe_valid(tmp_path: Any) -> None:
    p = tmp_path / "valid.txt"
    assert validate_path_safe(p) == p.resolve()


def test_validate_path_safe_traversal() -> None:
    with pytest.raises(ValueError, match="Path traversal attempt"):
        validate_path_safe(Path(".."))


def test_validate_path_safe_dangerous_chars() -> None:
    with pytest.raises(ValueError, match="Path contains invalid characters"):
        validate_path_safe(Path("file;rm"))

    with pytest.raises(ValueError, match="Path contains invalid characters"):
        validate_path_safe(Path("file|pipe"))

    with pytest.raises(ValueError, match="Path contains invalid characters"):
        validate_path_safe(Path("file*star"))

    with pytest.raises(ValueError, match="Path contains invalid characters"):
        validate_path_safe(Path("file{brace}"))


def test_validate_path_safe_flag_injection() -> None:
    with pytest.raises(ValueError, match="Filename cannot start with '-'"):
        validate_path_safe(Path("-rf"))


def test_validate_path_safe_outside_allowed() -> None:
    # /etc/passwd is outside allowed
    if Path("/etc/passwd").exists():
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_safe(Path("/etc/passwd"))
