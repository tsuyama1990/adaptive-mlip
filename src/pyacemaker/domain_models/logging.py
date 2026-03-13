from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LogMessagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    init_modules: str = "Initializing active learning modules"
    modules_init_success: str = "Core modules initialized successfully."
    module_init_fail: str = "Failed to initialize modules: {error}"
    start_loop: str = "Starting active learning loop"
    iteration_completed: str = "Active learning loop completed normally."
    potential_trained: str = "Training completed successfully."
    project_init: str = "PyAceMaker workflow started."
    start_iteration: str = "Starting iteration {iteration}/{max_iterations}"
    workflow_completed: str = "Workflow completed successfully."
    workflow_crashed: str = "Workflow failed with unrecoverable error: {error}"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level"
    )
    log_file: str | None = Field(default="pyacemaker.log", description="Path to the log file")
    max_bytes: int = Field(
        default=10 * 1024 * 1024, gt=0, description="Max size of log file before rotation"
    )
    messages: LogMessagesConfig = Field(
        default_factory=LogMessagesConfig, description="Configuration for specific log messages"
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
