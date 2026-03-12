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


class MDHaltInterrupt(Exception):  # noqa: N818
    """
    Exception thrown by the TwoTierEvaluator to cleanly halt the MD Engine.
    Contains the context of the physical anomaly.
    """

    def __init__(self, step: int, epicenter_indices: list[int]) -> None:
        self.step = step
        self.epicenter_indices = epicenter_indices
        msg = f"MD Halt triggered at step {step} with {len(epicenter_indices)} epicenter atoms."
        super().__init__(msg)
