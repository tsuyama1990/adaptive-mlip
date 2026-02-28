import contextlib

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AtomStructure(BaseModel):
    """
    A wrapper around ase.Atoms to carry metadata through the pipeline.
    Ensures strict typing and validation for energy, forces, stress, and uncertainty.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    atoms: Atoms = Field(..., description="The actual atomic structure")
    energy: float | None = Field(default=None, description="Potential energy of the structure (eV)")
    forces: np.ndarray | None = Field(
        default=None, description="Forces on atoms (eV/Angstrom), shape (N, 3)"
    )
    stress: np.ndarray | None = Field(
        default=None, description="Stress tensor (Voigt notation, 6 components) or full (3,3)"
    )
    uncertainty: float | None = Field(
        default=None, description="Model uncertainty metric (e.g., gamma)"
    )
    provenance: dict[str, str | int | float] = Field(
        default_factory=dict, description="Metadata about origin (e.g., {'step': 'direct_sampling'})"
    )

    @model_validator(mode="after")
    def validate_forces_shape(self) -> "AtomStructure":
        """
        Validates that the forces array shape matches the number of atoms.
        """
        if self.forces is not None:
            n_atoms = len(self.atoms)
            if self.forces.shape != (n_atoms, 3):
                msg = f"Forces shape mismatch: Expected ({n_atoms}, 3), got {self.forces.shape}"
                raise ValueError(msg)
        return self

    def to_ase(self) -> Atoms:
        """
        Returns a copy of the internal ase.Atoms object with all metadata attached
        to atoms.info and atoms.arrays (for forces).
        """
        atoms_copy: Atoms = self.atoms.copy() # type: ignore[no-untyped-call]

        if self.energy is not None:
            atoms_copy.info["energy"] = self.energy

        if self.stress is not None:
            atoms_copy.info["stress"] = self.stress

        if self.uncertainty is not None:
            atoms_copy.info["uncertainty"] = self.uncertainty
            atoms_copy.info["gamma"] = self.uncertainty # Alias for backward compatibility

        # Update provenance
        for key, value in self.provenance.items():
            atoms_copy.info[f"provenance_{key}"] = value

        if self.forces is not None:
            # Shape validation handled by model_validator
            atoms_copy.arrays["forces"] = self.forces

        return atoms_copy

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "AtomStructure":
        """
        Creates an AtomStructure from an ase.Atoms object, extracting metadata
        from info/arrays or calculator results if available.
        """
        return cls._extract_from_ase(atoms)

    @classmethod
    def _extract_from_ase(cls, atoms: Atoms) -> "AtomStructure":
        """Helper to reduce complexity of from_ase."""
        energy = None
        forces = None
        stress = None
        uncertainty = None
        provenance: dict[str, str | int | float] = {}

        # 1. Try to get from Calculator (High Priority)
        if atoms.calc:
            # Catch specific errors instead of bare Exception
            with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
                energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
            with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
                forces = atoms.get_forces() # type: ignore[no-untyped-call]
            with contextlib.suppress(PropertyNotImplementedError, RuntimeError):
                stress = atoms.get_stress() # type: ignore[no-untyped-call]

        # 2. Try to get from info/arrays (Low Priority / Override if Calc missing)
        if energy is None and "energy" in atoms.info:
             energy = atoms.info["energy"]

        if stress is None and "stress" in atoms.info:
             stress = atoms.info["stress"]

        if forces is None and "forces" in atoms.arrays:
             forces = atoms.arrays["forces"]

        # Uncertainty
        if "uncertainty" in atoms.info:
            uncertainty = atoms.info["uncertainty"]
        elif "gamma" in atoms.info:
            uncertainty = atoms.info["gamma"]

        # Provenance extraction
        for key, value in atoms.info.items():
            if key.startswith("provenance_"):
                clean_key = key.replace("provenance_", "")
                provenance[clean_key] = value

        return cls(
            atoms=atoms,
            energy=energy,
            forces=forces,
            stress=stress,
            uncertainty=uncertainty,
            provenance=provenance
        )
