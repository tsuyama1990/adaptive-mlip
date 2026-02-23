from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level")
    log_file: str | None = Field(default="pyacemaker.log", description="Path to the log file")
    max_bytes: int = Field(default=10 * 1024 * 1024, gt=0, description="Max size of log file before rotation")
    backup_count: int = Field(default=5, ge=0, description="Number of backup log files to keep")

    @field_validator("log_file")
    @classmethod
    def validate_log_file(cls, v: str | None) -> str | None:
        if v:
            path = Path(v)
            try:
                # Resolve to absolute path to check traversal
                abs_path = path.resolve()
                cwd = Path.cwd().resolve()
                if not abs_path.is_relative_to(cwd):
                    msg = f"Log file path must be inside the project directory: {v}"
                    raise ValueError(msg)
            except (ValueError, RuntimeError) as e:
                 # Re-raise nicely formatted
                 if isinstance(e, ValueError) and "must be inside" in str(e):
                     raise
                 msg = f"Invalid log file path: {e}"
                 raise ValueError(msg) from e

            if path.is_dir():
                msg = f"Log file path must be a file, not a directory: {v}"
                raise ValueError(msg)
        return v
