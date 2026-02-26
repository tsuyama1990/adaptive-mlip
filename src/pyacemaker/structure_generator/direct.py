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

    def update_config(self, config: Any) -> None:
        if not isinstance(config, StructureConfig):
            msg = "Expected StructureConfig"
            raise TypeError(msg)
        self.config = config

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

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

        # Validate elements uniqueness (Best Practice)
        if len(self.config.elements) != len(set(self.config.elements)):
            # This should be caught by Pydantic, but double check
            msg = "Duplicate elements found in configuration"
            raise ValueError(msg)

        # Validate r_cut (Best Practice)
        r_cut = getattr(self.config, "r_cut", 2.0)
        if r_cut <= 0:
            msg = "r_cut must be positive"
            raise ValueError(msg)

        # Create base template (cell definition)
        try:
            # Try to get a reasonable cell from the first element
            prim = bulk(self.config.elements[0])

            # Use supercell logic:
            # If user provided supercell_size, use it to scale the bulk.
            # But here we want a random gas.
            # If we repeat, we get N atoms.
            # We will use this number of atoms for random packing.

            atoms_template = prim.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

            # Scale cell volume to ensure reasonable density for random packing?
            # Random packing is inefficient. If density is high (bulk density), we will fail.
            # We should probably scale the cell up significantly to allow packing.
            # Let's scale volume by factor of 8 (2x in each dim) just for random gas generation
            # to make it easier, unless density is critical (but this is DIRECT sampling).

            cell = atoms_template.get_cell()
            atoms_template.set_cell(cell * 1.5, scale_atoms=False) # Expand box to ease packing

        except Exception:
            # Fallback to a 10x10x10 box if bulk fails
            # Log this if logging was available in this scope, for now proceed
            atoms_template = Atoms(
                self.config.elements[0],
                cell=[
                    DEFAULT_FALLBACK_CELL_SIZE,
                    DEFAULT_FALLBACK_CELL_SIZE,
                    DEFAULT_FALLBACK_CELL_SIZE,
                ],
                pbc=True,
            )
            # Scale supercell
            atoms_template = atoms_template.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]

        n_atoms = len(atoms_template)

        count = 0
        attempts = 0
        max_attempts = n_candidates * DEFAULT_GEN_MAX_ATTEMPTS_MULTIPLIER

        while count < n_candidates and attempts < max_attempts:
            attempts += 1
            candidate = atoms_template.copy()  # type: ignore[no-untyped-call]

            # Randomize positions (Uniform distribution in cell)
            # Use fractional coordinates [0, 1)
            scaled_pos = self._rng.uniform(0.0, 1.0, (n_atoms, 3))
            candidate.set_scaled_positions(scaled_pos)

            # Assign random elements if multiple are present
            if len(self.config.elements) > 1:
                symbols = self._rng.choice(self.config.elements, size=n_atoms)
                candidate.set_chemical_symbols(symbols)

            # Efficient Hard-Sphere Constraint (Scalability Fix)
            # Use ase.neighborlist.neighbor_list to find pairs within r_cut
            # "d" returns distances. If any distance is < r_cut (and > 0 for self), reject.
            # O(N) complexity for reasonable cutoffs.

            # Since neighbor_list returns all pairs (i, j) within cutoff,
            # if the list is not empty (excluding self-interaction if handled, but neighbor_list handles it),
            # then there is a clash.
            # We strictly want NO distance < r_cut.

            # Note: neighbor_list usually double counts (i-j and j-i).
            # If we find ANY pair with distance < r_cut, it's a clash.
            # However, for pure random packing, checking min distance via neighbor list is faster than all-pairs matrix for large N.

            # neighbor_list("d", atoms, cutoff) returns distances of neighbors.
            # If len(d) > 0, we have overlaps (since cutoff is r_cut).
            # Wait, neighbor_list excludes self-interaction by default? No, it depends.
            # But usually it returns actual neighbors.

            # Optimization:
            # If we request cutoff=r_cut/2 for sphere overlap? No, hard sphere means d_ij >= r_cut.
            # So if any d_ij < r_cut, fail.

            # For random gas, many atoms will overlap.
            # We check if *any* distance is less than r_cut.
            # Using neighbor_list with cutoff=r_cut

            # Note: For very high density, this loop might hang. max_attempts handles that.

            # neighbor_list("d", ...) returns distances.
            # However, for pure random packing, we can just use Atoms.get_all_distances() if N is small,
            # but neighbor_list is better for PBC.

            # Critical Fix: neighbor_list returns distances for ALL pairs within cutoff.
            # This includes self-interaction ONLY if self_interaction=True (default False).
            # So if len(distances) > 0, we have an overlap.

            # ISSUE: If r_cut is too large for the cell, neighbor_list might find periodic images.
            # That IS an overlap in PBC terms, so we should reject it.

            # HOWEVER, for random packing, maybe we just want to ensure atoms don't sit on top of each other.
            # If the density is too high, this loop will fail indefinitely.

            # Let's reduce r_cut slightly to allow packing or check if r_cut is realistic for the volume.
            # For now, we trust the config.

            # Fix: neighbor_list returns a tuple if we ask for more than one thing, but here we ask for "d".
            # It returns a single array.

            # Check for overlaps
            # Since r_cut is hard sphere diameter, we want d_ij > r_cut for all pairs.
            # neighbor_list("d", ...) returns all distances < r_cut.
            # So if len(distances) > 0, we have an overlap.

            # neighbor_list implementation detail:
            # It returns i, j, d. We just requested d.

            # To be absolutely robust against self-interaction (though neighbor_list usually handles it),
            # we can check indices.

            # Let's request indices too.
            indices_i, indices_j, dists = neighbor_list("ijd", candidate, r_cut)

            # Filter out self-interactions (i == j)
            mask = indices_i != indices_j
            real_overlaps = dists[mask]

            if len(real_overlaps) == 0:
                # Add metadata (Provenance)
                candidate.info["provenance"] = "DIRECT_SAMPLING"
                candidate.info["method"] = "random_packing"
                yield candidate
                count += 1
                attempts = 0  # Reset attempts on success

    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Not implemented for Cycle 01.
        """
        return iter([])
