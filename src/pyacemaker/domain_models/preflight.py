from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DiagnosticMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)

    node_id: str | None = Field(
        default=None, description="The ID of the DAG node where the issue originated"
    )
    severity: Severity = Field(..., description="Severity level of the diagnostic message")
    description: str = Field(..., description="Human-readable description of the issue")
    suggestion: str = Field(..., description="Actionable suggestion to resolve the issue")


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    errors: list[DiagnosticMessage] = Field(
        default_factory=list, description="Critical errors that block execution"
    )
    warnings: list[DiagnosticMessage] = Field(
        default_factory=list, description="Non-critical warnings that do not block execution"
    )
    info: list[DiagnosticMessage] = Field(
        default_factory=list, description="Informational messages confirming successful checks"
    )
