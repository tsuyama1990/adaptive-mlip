from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk

from pyacemaker.core.base import BaseGenerator
from pyacemaker.domain_models.structure import StructureConfig


class DirectSampler(BaseGenerator):
    """
    Generates random structures for initial exploration (Cycle 01).
    Implements random packing with hard-sphere constraints.
    """

    def __init__(self, config: StructureConfig) -> None:
        self.config = config

    def update_config(self, config: Any) -> None:
        if not isinstance(config, StructureConfig):
            msg = "Expected StructureConfig"
            raise TypeError(msg)
        self.config = config

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates n_candidates valid structures.
        """
        if n_candidates < 0:
            msg = "n_candidates must be non-negative"
            raise ValueError(msg)

        if n_candidates == 0:
            return

        if not self.config.elements:
            return

        # Create base template (cell definition)
        try:
            # Try to get a reasonable cell from the first element
            prim = bulk(self.config.elements[0])
            # Scale supercell
            atoms_template = prim.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]
        except Exception:
            # Fallback to a 10x10x10 box if bulk fails
            atoms_template = Atoms(
                self.config.elements[0], cell=[10.0, 10.0, 10.0], pbc=True
            )
            # Scale supercell
            atoms_template = atoms_template.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        n_atoms = len(atoms_template)
        r_cut = getattr(self.config, "r_cut", 2.0)

        count = 0
        attempts = 0
        # Limit attempts to avoid infinite loop
        max_attempts = n_candidates * 500

        while count < n_candidates and attempts < max_attempts:
            attempts += 1
            candidate = atoms_template.copy()  # type: ignore[no-untyped-call]

            # Randomize positions (Uniform distribution in cell)
            # Use fractional coordinates [0, 1)
            scaled_pos = np.random.uniform(0.0, 1.0, (n_atoms, 3))
            candidate.set_scaled_positions(scaled_pos)

            # Assign random elements if multiple are present
            if len(self.config.elements) > 1:
                symbols = np.random.choice(self.config.elements, size=n_atoms)
                candidate.set_chemical_symbols(symbols)

            # Check Hard-Sphere Constraint
            # O(N^2) check, acceptable for small initial systems
            # get_all_distances(mic=True) returns NxN matrix
            dists = candidate.get_all_distances(mic=True)  # type: ignore[no-untyped-call]

            # Set diagonal to infinity (self-distance)
            np.fill_diagonal(dists, np.inf)

            if np.min(dists) >= r_cut:
                yield candidate
                count += 1
                attempts = 0  # Reset attempts on success to keep going

    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Not implemented for Cycle 01.
        """
        return iter([])
