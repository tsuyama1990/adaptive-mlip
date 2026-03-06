class PyAceError(Exception):
    """Base exception for PYACEMAKER."""

class ConfigError(PyAceError):
    """Configuration related errors."""

class OracleError(PyAceError):
    """Oracle (DFT) related errors."""

class GeneratorError(PyAceError):
    """Structure generation errors."""

class TrainerError(PyAceError):
    """Training related errors."""

class EngineError(PyAceError):
    """MD Engine related errors."""

class ActiveSetError(PyAceError):
    """Active set selection errors."""

class OrchestratorError(PyAceError):
    """Orchestrator/Workflow related errors."""
