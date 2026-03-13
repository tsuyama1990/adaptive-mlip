from pyacemaker.domain_models.constants import (
    ERR_SIM_EXEC_FAIL,
    ERR_SIM_SECURITY_FAIL,
    ERR_SIM_SETUP_FAIL,
    ERR_SIM_UNEXPECTED,
)


class LammpsErrorHandler:
    """
    Translates errors from LAMMPS execution into domain-specific exceptions.
    """

    @staticmethod
    def handle(error: Exception) -> None:
        """
        Translates a generic exception into a specific PyAceMaker RuntimeError.

        Args:
            error: The exception to handle.

        Raises:
            RuntimeError: Domain-specific translated exception.
        """
        if isinstance(error, FileNotFoundError):
            raise RuntimeError(ERR_SIM_SETUP_FAIL.format(error=error)) from error  # noqa: TRY004
        if isinstance(error, ValueError):
            raise RuntimeError(ERR_SIM_SECURITY_FAIL.format(error=error)) from error  # noqa: TRY004
        if isinstance(error, RuntimeError):
            # If it's already translated or comes from inner layers wrapped as RuntimeError, re-raise appropriately
            if "Simulation execution failed" in str(error):
                raise error
            raise RuntimeError(ERR_SIM_EXEC_FAIL.format(error=error)) from error  # noqa: TRY004

        # Catch-all
        raise RuntimeError(ERR_SIM_UNEXPECTED.format(error=error)) from error
