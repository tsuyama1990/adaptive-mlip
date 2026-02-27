from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from pyacemaker.domain_models.active_learning import DescriptorConfig

from pyacemaker.domain_models.defaults import (
    DEFAULT_DESCRIPTOR_BATCH_SIZE,
    MAX_DESCRIPTOR_ARRAY_BYTES,
)

if TYPE_CHECKING:
    pass

class DescriptorCalculator:
    """
    Calculates descriptors for atomic structures using dscribe or other backends.

    This utility manages the complexity of interacting with descriptor libraries (like dscribe)
    and ensures efficient computation.

    Memory Implications:
    - Descriptors for large structures or high N/L max can be large.
    - Use `compute_stream` to yield batches and avoid OOM for large datasets.
    """
    def __init__(self, config: DescriptorConfig) -> None:
        self.config = config
        self._transformer = self._initialize_transformer()
        self._dim = self._estimate_dimension()

    def _initialize_transformer(self) -> object:
        """Initializes the descriptor transformer based on config."""
        method = self.config.method.lower()

        if method == "soap":
            from dscribe.descriptors import SOAP

            if self.config.sigma is None:
                raise ValueError("Sigma must be provided for SOAP descriptor")

            return SOAP(
                species=self.config.species,
                r_cut=self.config.r_cut,
                n_max=self.config.n_max,
                l_max=self.config.l_max,
                sigma=self.config.sigma,
                periodic=True, # Assume periodic structures for now
                sparse=self.config.sparse,
                average="inner" # Global descriptor
            )

        if method == "ace":
             raise NotImplementedError("ACE descriptor not yet implemented via dscribe wrapper")

        raise ValueError(f"Unknown descriptor method: {method}")

    def _estimate_dimension(self) -> int:
        """Estimates output dimension. Useful for pre-allocation or bounds checking."""
        method = self.config.method.lower()
        if method == "soap":
            return int(self._transformer.get_number_of_features()) # type: ignore[attr-defined]
        return 0

    def compute(self, atoms_list: list[Atoms], batch_size: int = DEFAULT_DESCRIPTOR_BATCH_SIZE) -> np.ndarray:
        """
        Computes descriptors for a list of Atoms objects using batching to prevent OOM.
        For very large datasets (>100k), use `compute_stream` instead to avoid holding
        all input Atoms and output descriptors in memory.

        Args:
            atoms_list: List of ASE Atoms objects.
            batch_size: Number of structures to process in memory at once.

        Returns:
            Numpy array of shape (N_structures, Descriptor_Dim).

        Raises:
            RuntimeError: If descriptor calculation fails.
            ValueError: If the total output size would exceed safe memory limits.
        """
        if not atoms_list:
            return np.array([])

        n_structures = len(atoms_list)

        # OOM Prevention Check BEFORE computation
        estimated_total_bytes = n_structures * self._dim * 8
        if estimated_total_bytes > MAX_DESCRIPTOR_ARRAY_BYTES:
            msg = f"Requested descriptor calculation total size exceeds safe memory limits ({estimated_total_bytes} > {MAX_DESCRIPTOR_ARRAY_BYTES} bytes). Use compute_stream instead."
            raise ValueError(msg)

        all_descriptors = []

        try:
            # Batch processing to prevent dscribe from consuming too much memory during intermediate steps
            for i in range(0, n_structures, batch_size):
                batch = atoms_list[i:i + batch_size]

                # dscribe's create method handles list of atoms
                descriptors = self._transformer.create(batch) # type: ignore[attr-defined]

                # Ensure it's a numpy array
                if not isinstance(descriptors, np.ndarray):
                    # If sparse, convert to dense for now
                    descriptors = descriptors.toarray()

                all_descriptors.append(descriptors)

            if all_descriptors:
                return np.vstack(all_descriptors)
            return np.array([])

        except Exception as e:
            raise RuntimeError(f"Descriptor calculation failed: {e}") from e

    def compute_stream(self, atoms_iter: Iterator[Atoms], batch_size: int = DEFAULT_DESCRIPTOR_BATCH_SIZE) -> Iterator[np.ndarray]:
        """
        Streams descriptor computation. Ideal for massive datasets to prevent OOM.
        Takes an iterator of Atoms and yields batches of descriptors as numpy arrays.

        Args:
            atoms_iter: Iterator yielding ASE Atoms objects.
            batch_size: Number of structures per batch.

        Yields:
            Numpy array of shape (batch_size, Descriptor_Dim).
        """
        batch: list[Atoms] = []
        try:
            for atoms in atoms_iter:
                batch.append(atoms)
                if len(batch) >= batch_size:
                    descriptors = self._transformer.create(batch) # type: ignore[attr-defined]
                    if not isinstance(descriptors, np.ndarray):
                        descriptors = descriptors.toarray()
                    yield descriptors
                    batch.clear()

            # Yield remaining
            if batch:
                descriptors = self._transformer.create(batch) # type: ignore[attr-defined]
                if not isinstance(descriptors, np.ndarray):
                    descriptors = descriptors.toarray()
                yield descriptors

        except Exception as e:
            raise RuntimeError(f"Streamed descriptor calculation failed: {e}") from e
