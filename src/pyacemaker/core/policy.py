from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from ase import Atoms

from pyacemaker.domain_models.defaults import DEFAULT_STRAIN_RANGE
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
            base_structure: The starting structure (unit cell).
            config: Configuration parameters including supercell size.
            n_structures: Number of structures to generate.
        Returns:
            Iterator yielding generated Atoms objects (supercells).
        """


class ColdStartPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Cold Start Policy: Returns the base structure repeated as a supercell.
        Yields exactly ONE structure regardless of n_structures (it's deterministic).
        """
        # Efficiently create the supercell only when yielding.
        yield base_structure.repeat(config.supercell_size)  # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Rattle Policy: Applies random displacement to atoms in the supercell.
        """
        # Create the supercell once. We must rattle the supercell, not unit cell.
        supercell = base_structure.repeat(config.supercell_size)  # type: ignore[no-untyped-call]

        for _ in range(n_structures):
            yield rattle(supercell, config.rattle_stdev, rng=self.rng)


class StrainPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Strain Policy: Applies random strain tensor to the cell.
        Optimization: Strains the unit cell first, THEN repeats to supercell.
        This is mathematically equivalent for homogeneous strain but more memory efficient
        during the strain operation itself (smaller matrices).
        """
        # Use configured magnitude or fallback
        magnitude = (
            config.strain_magnitude if config.strain_magnitude > 0 else DEFAULT_STRAIN_RANGE
        )

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

            # Apply strain to UNIT CELL
            strained_unit = apply_strain(base_structure, strain, rng=self.rng)

            # Then repeat to supercell
            yield strained_unit.repeat(config.supercell_size)  # type: ignore[no-untyped-call]


class DefectPolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1
    ) -> Iterator[Atoms]:
        """
        Defect Policy: Creates vacancies in the supercell.
        """
        # Defects must be created in the supercell to avoid periodicity artifacts
        supercell = base_structure.repeat(config.supercell_size)  # type: ignore[no-untyped-call]

        for _ in range(n_structures):
            yield create_vacancy(supercell, config.vacancy_rate, rng=self.rng)
