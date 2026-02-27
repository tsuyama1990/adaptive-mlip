from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from pyacemaker.domain_models.active_learning import DescriptorConfig
from pyacemaker.domain_models.defaults import MAX_DESCRIPTOR_ARRAY_BYTES

if TYPE_CHECKING:
    pass

class DescriptorCalculator:
    """
    Calculates descriptors for atomic structures using dscribe or other backends.

    This utility manages the complexity of interacting with descriptor libraries (like dscribe)
    and ensures efficient computation.

    Memory Implications:
    - Descriptors for large structures or high N/L max can be large.
    - Computation is batched to avoid OOM.
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
            # Dscribe SOAP global output size for average="inner"
            # Note: actual size might be slightly different depending on dscribe version,
            # so we just use transformer.get_number_of_features()
            return int(self._transformer.get_number_of_features())
        return 0

    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
        """
        Computes descriptors for a list of Atoms objects.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            Numpy array of shape (N_structures, Descriptor_Dim).

        Raises:
            RuntimeError: If descriptor calculation fails.
            ValueError: If input list is too large causing potential OOM.
        """
        if not atoms_list:
            return np.array([])

        # OOM Prevention Check
        n_structures = len(atoms_list)
        # Assuming float64 output (8 bytes)
        estimated_bytes = n_structures * self._dim * 8
        if estimated_bytes > MAX_DESCRIPTOR_ARRAY_BYTES:
            msg = f"Requested descriptor calculation exceeds safe memory limits ({estimated_bytes} > {MAX_DESCRIPTOR_ARRAY_BYTES} bytes). Use smaller batches."
            raise ValueError(msg)

        try:
            # dscribe's create method handles list of atoms
            # We already checked size, so it's safe to call
            descriptors = self._transformer.create(atoms_list)

            # Ensure it's a numpy array
            if not isinstance(descriptors, np.ndarray):
                # If sparse, convert to dense for now
                descriptors = descriptors.toarray()

            return descriptors

        except Exception as e:
            raise RuntimeError(f"Descriptor calculation failed: {e}") from e
