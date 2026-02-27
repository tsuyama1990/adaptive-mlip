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
        # Descriptor config is now mandatory
        self.descriptor_calc = DescriptorCalculator(self.config.descriptor)

    def _max_min_selection(self, descriptors: np.ndarray, n_select: int) -> list[int]:
        """
        Selects n_select indices using greedy MaxMin diversity algorithm.
        Supports memory-mapped arrays to process large datasets without OOM.

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

        selected_indices: list[int] = []
        remaining_indices = list(range(n_candidates))

        # 1. Select first point randomly
        first_idx = int(np.random.choice(remaining_indices))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Initialize min distances
        # If descriptors is a memmap, slicing it loads the slice into memory.
        # current_selected is (D,)
        current_selected = descriptors[first_idx]

        # Calculate initial distances. For memmap, we should do this in chunks to avoid allocating full (N_candidates,) in RAM?
        # Actually, a 1D array of float64 for 10M structures is 80MB. This easily fits in RAM.
        # The OOM risk is loading the (10M, D) descriptor matrix.
        # But `np.linalg.norm(descriptors - current_selected, axis=1)` creates an intermediate (10M, D) array in memory before reducing.
        # We MUST chunk the distance calculation to be memory safe.

        chunk_size = 10000
        min_dists = np.zeros(n_candidates, dtype=np.float64)

        for start_idx in range(0, n_candidates, chunk_size):
            end_idx = min(start_idx + chunk_size, n_candidates)
            chunk = descriptors[start_idx:end_idx]
            # Calculate distance for this chunk
            chunk_dists = np.linalg.norm(chunk - current_selected, axis=1)
            min_dists[start_idx:end_idx] = chunk_dists

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

            # Update min_dists using chunking
            new_selected = descriptors[next_idx]

            for start_idx in range(0, n_candidates, chunk_size):
                end_idx = min(start_idx + chunk_size, n_candidates)
                chunk = descriptors[start_idx:end_idx]
                chunk_dists = np.linalg.norm(chunk - new_selected, axis=1)

                # Update the specific chunk of min_dists
                min_dists[start_idx:end_idx] = np.minimum(
                    min_dists[start_idx:end_idx],
                    chunk_dists
                )

        return selected_indices

    def generate(self) -> Iterator[AtomStructure]:
        """
        Executes the sampling process using a batched approach to avoid OOM.
        Uses ASE SQLite database for efficient out-of-core storage of structures
        and numpy memmap for out-of-core storage of descriptors.
        """
        # Heuristic multiplier from config
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
                descs = self.descriptor_calc.compute(atoms_list)

                # Initialize memmap if first batch
                if memmap_array is None:
                    dim = descs.shape[1]
                    # Create memmap for the entire expected pool size
                    # Note: If gen_iter yields exactly n_pool, this perfectly fits.
                    # If it yields fewer, we'll slice it before MaxMin.
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
