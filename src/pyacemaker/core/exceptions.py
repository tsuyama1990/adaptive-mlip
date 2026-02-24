class PyAceError(Exception):
    """Base exception for all PyAceMaker errors."""


class OrchestratorError(PyAceError):
    """Raised when the workflow orchestration fails."""


class GeneratorError(PyAceError):
    """Raised when structure generation fails."""


class OracleError(PyAceError):
    """Raised when DFT calculation fails."""


class TrainerError(PyAceError):
    """Raised when potential training fails."""


class ActiveSetError(PyAceError):
    """Raised when active set selection fails."""


class LammpsDriverError(PyAceError):
    """Raised when LAMMPS execution fails."""


class ConfigError(PyAceError):
    """Raised when configuration is invalid."""


class EngineError(PyAceError):
    """Raised when MD engine execution fails."""
