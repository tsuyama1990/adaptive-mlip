from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


from pyacemaker.domain_models.defaults import (
    LOG_INIT_MODULES,
    LOG_ITERATION_COMPLETED,
    LOG_MODULE_INIT_FAIL,
    LOG_MODULES_INIT_SUCCESS,
)


class LogMessagesConfig(BaseModel):
    """Configurable log messages for the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    init_modules: str = Field(default=LOG_INIT_MODULES, description="Message for module init start")
    iteration_completed: str = Field(
        default=LOG_ITERATION_COMPLETED, description="Message for iteration completion"
    )
    module_init_fail: str = Field(
        default=LOG_MODULE_INIT_FAIL, description="Message for module init failure"
    )
    modules_init_success: str = Field(
        default=LOG_MODULES_INIT_SUCCESS, description="Message for successful module init"
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level"
    )
    messages: LogMessagesConfig = Field(
        default_factory=LogMessagesConfig, description="Configurable log messages"
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
                # If path does not exist, check if parent exists and is valid
                if path.exists():
                    abs_path = path.resolve(strict=True)
                else:
                    # Resolve parent directory strictly
                    parent = path.parent
                    if not parent.exists():
                        # Create if doesn't exist? No, config validation shouldn't create dirs
                        # But log setup might. Let's resolve what we can.
                        # For security, we ensure the parent is within CWD
                        pass

                    # Resolve as much as possible relative to CWD
                    abs_path = path.absolute().resolve(strict=False)

                cwd = Path.cwd().resolve(strict=True)
            except (ValueError, RuntimeError, OSError) as e:
                msg = f"Invalid log file path resolution: {e}"
                raise ValueError(msg) from e

            if not abs_path.is_relative_to(cwd):
                msg = f"Log file path must be inside the project directory: {v}"
                raise ValueError(msg)

        return v
