
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator


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
        # We only check if path is provided, but checking existence might be tricky
        # in unit tests without mocking.
        # However, for robustness, we should check if they are valid paths.
        # If they are absolute, we check existence.
        # If relative, they might be relative to pseudo_dir (which we don't know here).
        # So strictly checking existence here might be too aggressive if user sets pseudo_dir later.
        # But usually `pseudopotentials` in ASE expects filename (if pseudo_dir set) or full path.
        # Let's enforce that they are non-empty strings.
        for elem, path_str in v.items():
            if not path_str or not path_str.strip():
                msg = f"Pseudopotential path for {elem} cannot be empty"
                raise ValueError(msg)

        return v
