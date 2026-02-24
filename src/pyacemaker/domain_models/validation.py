from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon_supercell: list[int] = Field(
        default=[2, 2, 2],
        description="Supercell matrix for phonon calculations (e.g., [2, 2, 2])"
    )
    phonon_displacement: PositiveFloat = Field(
        default=0.01,
        description="Displacement distance for finite difference phonon calculation (Angstrom)"
    )
    elastic_strain: PositiveFloat = Field(
        default=0.01,
        description="Maximum strain for elastic constant calculation (fraction)"
    )
    imaginary_frequency_tolerance: float = Field(
        default=-0.05,
        description="Tolerance for imaginary frequencies (THz). Values below this are considered unstable."
    )
    symprec: PositiveFloat = Field(
        default=1e-5,
        description="Symmetry precision for structure analysis (Angstrom)"
    )

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phonon_stable: bool = Field(..., description="Whether the structure is dynamically stable")
    elastic_stable: bool = Field(..., description="Whether the structure is mechanically stable")
    imaginary_frequencies: list[float] = Field(..., description="List of imaginary frequencies detected (THz)")
    elastic_tensor: list[list[float]] = Field(..., description="Elastic stiffness tensor (Cij) in GPa")
    bulk_modulus: float = Field(..., description="Bulk modulus (Voigt-Reuss-Hill average) in GPa")
    shear_modulus: float = Field(..., description="Shear modulus (Voigt-Reuss-Hill average) in GPa")
    plots: dict[str, str] = Field(
        default_factory=dict,
        description="Base64 encoded plots (e.g., 'band_structure', 'dos')"
    )
