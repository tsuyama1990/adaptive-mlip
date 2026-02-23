from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator

from pyacemaker.constants import DEFAULT_MIXING_BETA_FACTOR, DEFAULT_SMEARING_WIDTH_FACTOR


class DFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use")
    functional: str = Field(..., description="Exchange-correlation functional")
    kpoints_density: PositiveFloat = Field(..., description="K-points density in 1/Angstrom")
    encut: PositiveFloat = Field(..., description="Energy cutoff in eV")

    # Self-healing and convergence parameters
    mixing_beta: float = Field(0.7, gt=0.0, le=1.0, description="Initial mixing parameter for SCF")
    smearing_type: str = Field("mv", description="Type of smearing (e.g., 'mv', 'gaussian')")
    smearing_width: PositiveFloat = Field(0.1, description="Width of smearing in eV")
    diagonalization: str = Field("david", description="Diagonalization algorithm")

    # Strategy Multipliers
    mixing_beta_factor: float = Field(DEFAULT_MIXING_BETA_FACTOR, gt=0.0, le=1.0, description="Multiplier for mixing_beta reduction strategy")
    smearing_width_factor: float = Field(DEFAULT_SMEARING_WIDTH_FACTOR, gt=1.0, description="Multiplier for smearing_width increase strategy")

    # Pseudopotentials
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping of element symbols to pseudopotential filenames"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """
        Validates that pseudopotential files exist.
        """
        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

            p = Path(path_str)
            # If path is absolute, check existence
            if p.is_absolute() and not p.exists():
                msg = f"Pseudopotential file not found: {p}"
                raise FileNotFoundError(msg)

            # Simple traversal check for relative paths
            if ".." in str(p):
                 msg = f"Relative path traversal (..) not allowed in pseudopotentials: {p}"
                 raise ValueError(msg)

        return v
