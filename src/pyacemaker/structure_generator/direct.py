from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from pyacemaker.core.base import BaseGenerator
from pyacemaker.domain_models.constants import (
    DEFAULT_FALLBACK_CELL_SIZE,
    DEFAULT_GEN_MAX_ATTEMPTS_MULTIPLIER,
    DEFAULT_VOLUME_SCALING_FACTOR,
)
from pyacemaker.domain_models.structure import StructureConfig


class DirectSampler(BaseGenerator):
    """
    Generates random structures for initial exploration (Cycle 01).
    Implements random packing with hard-sphere constraints.
    """

    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng()

    def update_config(self, config: StructureConfig) -> None:
        if not isinstance(config, StructureConfig):
            msg = "Expected StructureConfig"
            raise TypeError(msg)
        self.config = config

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def _create_template(self) -> Atoms:
        """Creates the template atoms object and validates feasibility."""
        r_cut = self.config.r_cut
        try:
            prim = bulk(self.config.elements[0])
            atoms_template = prim.repeat(self.config.supercell_size) # type: ignore[no-untyped-call]
            cell = atoms_template.get_cell()
            atoms_template.set_cell(cell * DEFAULT_VOLUME_SCALING_FACTOR, scale_atoms=False)

            # Validation
            vol = atoms_template.get_volume() # type: ignore[no-untyped-call]
            n_atoms_check = len(atoms_template)
            sphere_vol = (4/3) * np.pi * ((r_cut / 2.0) ** 3)
            packing_fraction = (n_atoms_check * sphere_vol) / vol

            if packing_fraction > 0.55:
                msg = f"Impossible packing density: {packing_fraction:.2f} > 0.55. Increase supercell size or decrease r_cut."
                raise ValueError(msg)

        except ValueError:
             # Reraise ValueError from packing check
             raise
        except Exception as e:
            # Fallback for ASE build errors (e.g. crystal structure generation fail)
            # Only if it's NOT our packing density error
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create template from bulk: {e}. Using fallback cubic cell.")

            atoms_template = Atoms(
                self.config.elements[0],
                cell=[DEFAULT_FALLBACK_CELL_SIZE, DEFAULT_FALLBACK_CELL_SIZE, DEFAULT_FALLBACK_CELL_SIZE],
                pbc=True,
            )
            # Ensure cell is properly set before repeating
            if not np.any(atoms_template.get_cell()):
                 # If cell is 0, something went wrong with init.
                 # Force set cell.
                 atoms_template.set_cell([DEFAULT_FALLBACK_CELL_SIZE]*3)

            atoms_template = atoms_template.repeat(self.config.supercell_size) # type: ignore[no-untyped-call]

        return atoms_template

    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates n_candidates valid structures.
        """
        if n_candidates < 0:
            msg = "n_candidates must be non-negative"
            raise ValueError(msg)

        if n_candidates == 0 or not self.config.elements:
            return

        if len(self.config.elements) != len(set(self.config.elements)):
            msg = "Duplicate elements found in configuration"
            raise ValueError(msg)

        # Validate chemical symbols
        from ase.data import chemical_symbols
        valid_set = set(chemical_symbols)
        for el in self.config.elements:
            if el not in valid_set:
                msg = f"Invalid chemical symbol: {el}"
                raise ValueError(msg)

        r_cut = getattr(self.config, "r_cut", 2.0)
        if r_cut <= 0:
            msg = "r_cut must be positive"
            raise ValueError(msg)

        atoms_template = self._create_template()
        n_atoms = len(atoms_template)

        count = 0
        attempts = 0
        max_attempts = n_candidates * DEFAULT_GEN_MAX_ATTEMPTS_MULTIPLIER

        while count < n_candidates and attempts < max_attempts:
            attempts += 1

            scaled_pos = self._rng.uniform(0.0, 1.0, (n_atoms, 3))
            atoms_template.set_scaled_positions(scaled_pos)

            if len(self.config.elements) > 1:
                symbols = self._rng.choice(self.config.elements, size=n_atoms)
                atoms_template.set_chemical_symbols(symbols)

            indices_i, indices_j, dists = neighbor_list("ijd", atoms_template, r_cut)
            mask = indices_i != indices_j

            if len(dists[mask]) == 0:
                # IMPORTANT: We must yield a copy because atoms_template is mutated in the next iteration.
                # To minimize memory churn, we could conceptually yield the arrays and let consumer construct,
                # but BaseGenerator contract requires yielding Atoms objects.
                # We stick to copy() for safety, but ensure we don't hold references to it internally.
                candidate = atoms_template.copy() # type: ignore[no-untyped-call]
                candidate.info["provenance"] = "DIRECT_SAMPLING"
                candidate.info["method"] = "random_packing"
                yield candidate
                count += 1
                attempts = 0

    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Not implemented for Cycle 01.
        """
        return iter([])
