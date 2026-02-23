from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from ase import Atoms

from pyacemaker.constants import DEFAULT_STRAIN_RANGE
from pyacemaker.domain_models.structure import StrainMode, StructureConfig
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


class BasePolicy(ABC):
    def __init__(self) -> None:
        # Initialize RNG once per policy instance
        self.rng = np.random.default_rng()

    @abstractmethod
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Generate structures based on the policy.
        Args:
            base_structure: The starting structure (pristine).
            config: Configuration parameters.
            n_structures: Number of structures to generate.
        Returns:
            Iterator yielding generated Atoms objects.
        """


class ColdStartPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Cold Start Policy: Returns the base structure as is.
        """
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Rattle Policy: Applies random displacement to atoms.
        """
        for _ in range(n_structures):
            yield rattle(base_structure, config.rattle_stdev, rng=self.rng)


class StrainPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Strain Policy: Applies random strain tensor to the cell.
        """
        # Use configured magnitude or fallback
        magnitude = config.strain_magnitude if config.strain_magnitude > 0 else DEFAULT_STRAIN_RANGE

        for _ in range(n_structures):
            strain = np.zeros((3, 3))

            if config.strain_mode == StrainMode.VOLUME:
                # Hydrostatic strain (uniform scaling)
                val = self.rng.uniform(-magnitude, magnitude)
                np.fill_diagonal(strain, val)

            elif config.strain_mode == StrainMode.SHEAR:
                # Pure shear (off-diagonal elements)
                s12 = self.rng.uniform(-magnitude, magnitude)
                s13 = self.rng.uniform(-magnitude, magnitude)
                s23 = self.rng.uniform(-magnitude, magnitude)
                strain[0, 1] = strain[1, 0] = s12
                strain[0, 2] = strain[2, 0] = s13
                strain[1, 2] = strain[2, 1] = s23

            else:
                # Mixed: Random symmetric tensor
                rand = self.rng.uniform(-magnitude, magnitude, (3, 3))
                strain = (rand + rand.T) / 2

            yield apply_strain(base_structure, strain, rng=self.rng)


class DefectPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Defect Policy: Creates vacancies.
        """
        for _ in range(n_structures):
            yield create_vacancy(base_structure, config.vacancy_rate, rng=self.rng)
