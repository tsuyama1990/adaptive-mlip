from collections.abc import Iterator

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

        # Initialize min distances to selected set (initially just first point)
        # distance from first point to all others
        # using Euclidean distance
        # Shape: (N_candidates,)

        # Compute distances from the first selected point to all candidates
        # Optimization: maintain min_dist array
        current_selected = descriptors[first_idx]
        min_dists = np.linalg.norm(descriptors - current_selected, axis=1)

        # 2. Greedy selection
        for _ in range(n_select - 1):
            if not remaining_indices:
                break

            # Find point in remaining_indices with maximum min_dist
            # We can just look at min_dists array, but mask out already selected (set to -1)
            # Or just argmax on remaining

            # Efficient way:
            # min_dists contains min(d(x, s) for s in selected) for all x
            # We want x that maximizes this value.

            # Get argmax only among remaining
            # But argmax over subset is tricky with indices.
            # Let's just iterate n_select times.

            # Pick candidate with largest minimum distance
            # Setting selected distances to -1 to ignore them
            valid_min_dists = min_dists.copy()
            valid_min_dists[selected_indices] = -1.0

            next_idx = int(np.argmax(valid_min_dists))

            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

            # Update min_dists
            # New min_dist is min(old_min_dist, dist_to_newly_selected)
            new_selected = descriptors[next_idx]
            dists_to_new = np.linalg.norm(descriptors - new_selected, axis=1)
            min_dists = np.minimum(min_dists, dists_to_new)

        return selected_indices

    def generate(self) -> Iterator[AtomStructure]:
        """
        Executes the sampling process:
        1. Generate a large pool of candidates.
        2. Compute descriptors.
        3. Select diverse subset.
        4. Yield selected structures.
        """
        # 1. Generate Candidates
        # Heuristic: Generate 10x candidates to sample from
        n_pool = self.config.target_points * 10
        logger.info(f"Generating candidate pool of size {n_pool} for DIRECT sampling...")

        # Generator returns Iterator, consume it to list for descriptor calc
        # Note: This loads all candidates into memory. For N=1000 or 10000 this is fine.
        candidates = list(self.generator.generate(n_candidates=n_pool))

        if not candidates:
            logger.warning("Generator produced no candidates.")
            return

        # Extract ASE atoms for descriptor calculation
        atoms_list = [c.atoms for c in candidates]

        # 2. Compute Descriptors
        logger.info("Computing global descriptors...")
        descriptors = self.descriptor_calc.compute(atoms_list)

        # 3. Select Diverse Subset
        logger.info(f"Selecting {self.config.target_points} most diverse structures...")
        selected_indices = self._max_min_selection(descriptors, self.config.target_points)

        # 4. Yield Results
        for idx in selected_indices:
            structure = candidates[idx]
            structure.provenance["sampling_method"] = "direct_maxmin"
            structure.provenance["descriptor"] = self.config.descriptor.method
            yield structure
