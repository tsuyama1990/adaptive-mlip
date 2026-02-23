from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from ase import Atoms

from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle

DEFAULT_STRAIN_RANGE = 0.05


class BasePolicy(ABC):
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
        Typically used when we just want the initial guess (e.g. from M3GNet) without perturbation.
        We yield it once (or N times if requested? Cold Start usually implies 1 unique structure).
        However, to satisfy the interface, if n > 1 is requested, we might yield it N times?
        But duplicates are useless.
        Let's yield it ONCE regardless of n_structures, or maybe up to n?
        For Cold Start, yielding 1 is the intended behavior.
        The caller can handle replication if needed.
        """
        # Yield once.
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Rattle Policy: Applies random displacement to atoms.
        """
        for _ in range(n_structures):
            yield rattle(base_structure, config.rattle_stdev)


class StrainPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Strain Policy: Applies random strain tensor to the cell.
        """
        magnitude = DEFAULT_STRAIN_RANGE

        for _ in range(n_structures):
            strain = np.zeros((3, 3))

            if config.strain_mode == "volume":
                # Hydrostatic strain (uniform scaling)
                val = np.random.uniform(-magnitude, magnitude)
                np.fill_diagonal(strain, val)

            elif config.strain_mode == "shear":
                # Pure shear (off-diagonal elements)
                # We apply random symmetric shear
                s12 = np.random.uniform(-magnitude, magnitude)
                s13 = np.random.uniform(-magnitude, magnitude)
                s23 = np.random.uniform(-magnitude, magnitude)
                strain[0, 1] = strain[1, 0] = s12
                strain[0, 2] = strain[2, 0] = s13
                strain[1, 2] = strain[2, 1] = s23

            else:
                # Mixed: Random symmetric tensor
                # We generate a random matrix and symmetrize it
                rand = np.random.uniform(-magnitude, magnitude, (3, 3))
                strain = (rand + rand.T) / 2

            yield apply_strain(base_structure, strain)


class DefectPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Defect Policy: Creates vacancies.
        """
        for _ in range(n_structures):
            yield create_vacancy(base_structure, config.vacancy_rate)
