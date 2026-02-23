from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level"
    )
    log_file: str | None = Field(default="pyacemaker.log", description="Path to the log file")
    max_bytes: int = Field(
        default=10 * 1024 * 1024, gt=0, description="Max size of log file before rotation"
    )
    backup_count: int = Field(default=5, ge=0, description="Number of backup log files to keep")

    @field_validator("log_file")
    @classmethod
    def validate_log_file(cls, v: str | None) -> str | None:
        if v:
            path = Path(v)
            if path.is_dir():
                msg = f"Log file path must be a file, not a directory: {v}"
                raise ValueError(msg)

            try:
                # Use strict resolve to follow symlinks and check traversal
                abs_path = path.resolve(strict=False)
                cwd = Path.cwd().resolve(strict=True)
            except (ValueError, RuntimeError, OSError) as e:
                msg = f"Invalid log file path resolution: {e}"
                raise ValueError(msg) from e

            if not abs_path.is_relative_to(cwd):
                msg = f"Log file path must be inside the project directory: {v}"
                raise ValueError(msg)

        return v
