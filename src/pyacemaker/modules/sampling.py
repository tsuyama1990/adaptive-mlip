import tempfile
from collections.abc import Iterator
from pathlib import Path

import ase.db
import numpy as np

from pyacemaker.core.base import BaseGenerator
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig
from pyacemaker.logger import get_logger
from pyacemaker.utils.descriptors import DescriptorCalculator

logger = get_logger()

class DirectSampler:
    """
    Implements DIRECT sampling (entropy maximization) for structure selection.
    """
    def __init__(self, config: Step1DirectSamplingConfig, generator: BaseGenerator) -> None:
        """
        Args:
            config: Configuration for direct sampling.
            generator: Base structure generator to produce candidates.
        """
        self.config = config
        self.generator = generator
        self.descriptor_calc = DescriptorCalculator(config.descriptor)

    def _max_min_selection(self, descriptors: np.ndarray, n_select: int) -> list[int]:
        """
        Selects n_select indices using greedy MaxMin diversity algorithm.

        Args:
            descriptors: (N_candidates, D) numpy array.
            n_select: Number of points to select.

        Returns:
            List of selected indices.
        """
        n_candidates = descriptors.shape[0]
        if n_select > n_candidates:
            logger.warning(f"Requested {n_select} samples but only {n_candidates} available. Selecting all.")
            return list(range(n_candidates))

        selected_indices: list[int] = []
        remaining_indices = list(range(n_candidates))

        # 1. Select first point randomly
        first_idx = int(np.random.choice(remaining_indices))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Initialize min distances
        current_selected = descriptors[first_idx]
        min_dists = np.linalg.norm(descriptors - current_selected, axis=1)

        # 2. Greedy selection
        for _ in range(n_select - 1):
            if not remaining_indices:
                break

            valid_min_dists = min_dists.copy()
            # Mask selected
            valid_min_dists[selected_indices] = -1.0

            next_idx = int(np.argmax(valid_min_dists))

            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

            # Update min_dists
            new_selected = descriptors[next_idx]
            dists_to_new = np.linalg.norm(descriptors - new_selected, axis=1)
            min_dists = np.minimum(min_dists, dists_to_new)

        return selected_indices

    def generate(self) -> Iterator[AtomStructure]:
        """
        Executes the sampling process using a batched approach to avoid OOM.
        Uses ASE SQLite database for efficient out-of-core storage.
        """
        # Heuristic: 10x candidates
        n_pool = self.config.target_points * 10
        batch_size = self.config.batch_size

        logger.info(f"Generating candidate pool of size {n_pool} for DIRECT sampling...")

        all_descriptors = []

        # Use temp sqlite db to store candidates
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            db = ase.db.connect(str(tmp_path)) # type: ignore[no-untyped-call]

            # Pass 1: Generate + Compute Descriptors + Cache Structures in SQLite
            gen_iter = self.generator.generate(n_candidates=n_pool)

            current_batch: list[AtomStructure] = []

            def process_batch(batch: list[AtomStructure]) -> None:
                if not batch:
                    return
                # Write to temp db
                atoms_list = [s.to_ase() for s in batch]
                for atoms in atoms_list:
                    # Storing atoms natively in ase.db. Information/provenance might need
                    # specific mapping or we reconstruct later.
                    db.write(atoms) # type: ignore[no-untyped-call]

                # Compute descriptors
                descs = self.descriptor_calc.compute(atoms_list)
                all_descriptors.append(descs)

            for struct in gen_iter:
                current_batch.append(struct)
                if len(current_batch) >= batch_size:
                    process_batch(current_batch)
                    current_batch = []

            # Process remaining
            process_batch(current_batch)

            if not all_descriptors:
                logger.warning("No candidates generated.")
                return

            # Combine descriptors
            full_descriptor_matrix = np.vstack(all_descriptors)

            # Pass 2: Select Indices
            logger.info(f"Selecting {self.config.target_points} structures from pool of {full_descriptor_matrix.shape[0]}...")
            selected_indices = set(self._max_min_selection(full_descriptor_matrix, self.config.target_points))

            # Pass 3: Retrieve selected from DB
            # ASE db IDs are 1-indexed. Our selected_indices are 0-indexed.
            for idx in selected_indices:
                db_id = idx + 1
                row = db.get(id=db_id) # type: ignore[no-untyped-call]
                atoms = row.toatoms() # type: ignore[no-untyped-call]

                # Reconstruct AtomStructure
                structure = AtomStructure.from_ase(atoms)

                structure.provenance["sampling_method"] = "direct_maxmin"
                structure.provenance["descriptor"] = self.config.descriptor.method
                yield structure

        finally:
            # Cleanup temp file
            if tmp_path.exists():
                tmp_path.unlink()
