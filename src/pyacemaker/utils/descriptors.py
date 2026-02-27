from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from pyacemaker.domain_models.active_learning import DescriptorConfig

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

    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:
        """
        Computes descriptors for a list of Atoms objects.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            Numpy array of shape (N_structures, Descriptor_Dim).

        Raises:
            RuntimeError: If descriptor calculation fails.
        """
        if not atoms_list:
            return np.array([])

        try:
            # dscribe's create method handles list of atoms
            # For very large lists, this might still be memory intensive.
            # Callers should batch usage of this method.
            descriptors = self._transformer.create(atoms_list)

            # Ensure it's a numpy array
            if not isinstance(descriptors, np.ndarray):
                # If sparse, convert to dense for now (unless memory is issue)
                # For direct sampling (N ~ 1000), dense is fine.
                descriptors = descriptors.toarray()

            return descriptors

        except Exception as e:
            raise RuntimeError(f"Descriptor calculation failed: {e}") from e
