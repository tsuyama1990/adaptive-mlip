import tempfile
from collections.abc import Iterator
from pathlib import Path

import ase.db
import numpy as np

from pyacemaker.core.base import BaseGenerator
from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.defaults import MAX_MEMMAP_CHUNK_SIZE
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
        # Descriptor config is now mandatory
        self.descriptor_calc = DescriptorCalculator(self.config.descriptor)

    def _max_min_selection(self, descriptors: np.ndarray, n_select: int) -> list[int]:
        """
        Selects n_select indices using greedy MaxMin diversity algorithm.
        Supports memory-mapped arrays to process large datasets without OOM.

        For very large N (e.g., > 100k), MaxMin is approximated by sub-sampling
        to ensure reasonable execution time, as O(k*N) can be slow for large D and N.

        Args:
            descriptors: (N_candidates, D) numpy array or memmap.
            n_select: Number of points to select.

        Returns:
            List of selected indices.
        """
        n_candidates = descriptors.shape[0]
        if n_select > n_candidates:
            logger.warning(f"Requested {n_select} samples but only {n_candidates} available. Selecting all.")
            return list(range(n_candidates))

        # Approximation for very large datasets to bound execution time
        MAX_EXACT_CANDIDATES = 100_000
        if n_candidates > MAX_EXACT_CANDIDATES:
            logger.info(f"Dataset too large for exact MaxMin ({n_candidates} > {MAX_EXACT_CANDIDATES}). Subsampling...")
            # Randomly select a subset to perform MaxMin on
            # This is a standard fast approximation for Farthest Point Sampling
            subset_indices = np.random.choice(n_candidates, MAX_EXACT_CANDIDATES, replace=False)
            # Create a new memmap or just in-memory if D is small enough
            # But to be safe with OOM, we just map indices
            active_indices = list(subset_indices)
        else:
            active_indices = list(range(n_candidates))

        n_active = len(active_indices)

        # We need a mapping from active_index position -> actual descriptor index
        # active_indices[i] = real_index

        selected_real_indices: list[int] = []

        # 1. Select first point randomly
        first_active_pos = int(np.random.choice(n_active))
        first_real_idx = active_indices[first_active_pos]
        selected_real_indices.append(first_real_idx)

        # Remove from active pool (swap with last for O(1) removal, or just use boolean mask)
        # Using a boolean mask is faster for large arrays
        is_active = np.ones(n_active, dtype=bool)
        is_active[first_active_pos] = False

        # Initialize min distances
        current_selected = descriptors[first_real_idx]

        # Calculate initial squared distances to avoid intermediate array allocation overhead
        chunk_size = MAX_MEMMAP_CHUNK_SIZE
        # We store min_dists only for the active_indices subset
        min_dists = np.zeros(n_active, dtype=np.float64)

        # Compute initial distances in chunks over the active subset
        for start_pos in range(0, n_active, chunk_size):
            end_pos = min(start_pos + chunk_size, n_active)
            # Get real indices for this chunk
            chunk_real_indices = active_indices[start_pos:end_pos]
            # Fetch from memmap (fancy indexing on memmap might load into memory, but it's bounded by chunk_size)
            chunk = descriptors[chunk_real_indices]
            # Use squared distance
            chunk_dists = np.sum((chunk - current_selected)**2, axis=1)
            min_dists[start_pos:end_pos] = chunk_dists

        # 2. Greedy selection
        for _ in range(n_select - 1):
            # Mask out already selected
            valid_min_dists = min_dists.copy()
            valid_min_dists[~is_active] = -1.0

            # Find next point
            next_active_pos = int(np.argmax(valid_min_dists))
            next_real_idx = active_indices[next_active_pos]

            selected_real_indices.append(next_real_idx)
            is_active[next_active_pos] = False

            # Update min_dists using chunking
            new_selected = descriptors[next_real_idx]

            for start_pos in range(0, n_active, chunk_size):
                end_pos = min(start_pos + chunk_size, n_active)
                chunk_real_indices = active_indices[start_pos:end_pos]
                chunk = descriptors[chunk_real_indices]
                chunk_dists = np.sum((chunk - new_selected)**2, axis=1)

                # Update the specific chunk of min_dists
                min_dists[start_pos:end_pos] = np.minimum(
                    min_dists[start_pos:end_pos],
                    chunk_dists
                )

        return selected_real_indices

    def generate(self) -> Iterator[AtomStructure]:
        """
        Executes the sampling process using a batched approach to avoid OOM.
        Uses ASE SQLite database for efficient out-of-core storage of structures
        and numpy memmap for out-of-core storage of descriptors.
        """
        n_pool = self.config.target_points * self.config.candidate_multiplier
        batch_size = self.config.batch_size

        logger.info(f"Generating candidate pool of size {n_pool} for DIRECT sampling...")

        # Use temp sqlite db to store candidates
        tmp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db_path = Path(tmp_db_file.name)
        tmp_db_file.close()

        # Use temp file for memory-mapped descriptors
        tmp_memmap_file = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_memmap_path = Path(tmp_memmap_file.name)
        tmp_memmap_file.close()

        try:
            db = ase.db.connect(str(tmp_db_path)) # type: ignore[no-untyped-call]

            # Pass 1: Generate + Compute Descriptors + Cache Structures
            gen_iter = self.generator.generate(n_candidates=n_pool)

            current_batch: list[AtomStructure] = []

            # Since we stream, we don't know the descriptor dimension until the first batch.
            # We initialize memmap lazily.
            memmap_array = None
            current_row_idx = 0

            def process_batch(batch: list[AtomStructure]) -> None:
                nonlocal memmap_array, current_row_idx
                if not batch:
                    return
                # Write to temp db
                atoms_list = [s.to_ase() for s in batch]
                for atoms in atoms_list:
                    # Storing atoms natively in ase.db.
                    db.write(atoms) # type: ignore[no-untyped-call]

                # Compute descriptors
                descs = self.descriptor_calc.compute(atoms_list, batch_size=batch_size)

                # Initialize memmap if first batch
                if memmap_array is None:
                    dim = descs.shape[1]
                    # Create memmap for the entire expected pool size
                    memmap_array = np.memmap(
                        str(tmp_memmap_path),
                        dtype='float64',
                        mode='w+',
                        shape=(n_pool, dim)
                    )

                # Write to memmap
                n_descs = descs.shape[0]
                memmap_array[current_row_idx:current_row_idx + n_descs] = descs
                current_row_idx += n_descs

            for struct in gen_iter:
                current_batch.append(struct)
                if len(current_batch) >= batch_size:
                    process_batch(current_batch)
                    current_batch = []

            # Process remaining
            process_batch(current_batch)

            if current_row_idx == 0 or memmap_array is None:
                logger.warning("No candidates generated.")
                return

            # Ensure data is flushed
            memmap_array.flush()

            # If the generator returned fewer than expected, slice the memmap
            # to avoid trailing zeros. Read-only mode for safety.
            valid_descriptors = np.memmap(
                str(tmp_memmap_path),
                dtype='float64',
                mode='r',
                shape=(current_row_idx, memmap_array.shape[1])
            )

            # Pass 2: Select Indices
            logger.info(f"Selecting {self.config.target_points} structures from pool of {valid_descriptors.shape[0]}...")
            selected_indices = self._max_min_selection(valid_descriptors, self.config.target_points)

            # Pass 3: Retrieve selected from DB
            db_ids = [idx + 1 for idx in selected_indices]

            selected_set = set(db_ids)

            # Iterate through DB once (streaming) and yield matches
            for row in db.select(): # type: ignore[no-untyped-call]
                if row.id in selected_set: # type: ignore[attr-defined]
                    atoms = row.toatoms() # type: ignore[no-untyped-call]
                    structure = AtomStructure.from_ase(atoms)
                    structure.provenance["sampling_method"] = "direct_maxmin"
                    structure.provenance["descriptor"] = self.config.descriptor.method
                    yield structure

        finally:
            # Cleanup temp files
            if tmp_db_path.exists():
                tmp_db_path.unlink()
            if tmp_memmap_path.exists():
                tmp_memmap_path.unlink()
