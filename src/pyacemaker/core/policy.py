import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from pyacemaker.domain_models.constants import DEFAULT_STRAIN_RANGE
from pyacemaker.domain_models.structure import StrainMode, StructureConfig
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle

logger = logging.getLogger(__name__)

class BasePolicy(ABC):
    def __init__(self) -> None:
        # Initialize RNG once per policy instance
        self.rng = np.random.default_rng()

    @abstractmethod
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generate structures based on the policy.
        Args:
            base_structure: The starting structure (pristine).
            config: Configuration parameters.
            n_structures: Number of structures to generate.
            **kwargs: Additional arguments (e.g., engine).
        Returns:
            Iterator yielding generated Atoms objects.
        """


class ColdStartPolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Cold Start Policy: Returns the base structure as is.
        """
        # Cold start yields only one structure (the base) regardless of n_structures.
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Rattle Policy: Applies random displacement to atoms.
        """
        for _ in range(n_structures):
            yield rattle(base_structure, config.rattle_stdev, rng=self.rng)


class StrainPolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
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
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Defect Policy: Creates vacancies.
        """
        for _ in range(n_structures):
            yield create_vacancy(base_structure, config.vacancy_rate, rng=self.rng)


class NormalModePolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Normal Mode Policy: Perturbs along normal modes.
        For now, falls back to rattle with warning if engine not provided or expensive calculation skipped.
        """
        # Placeholder for full implementation
        # Real implementation would calculate Hessian via kwargs['engine']
        for _ in range(n_structures):
            yield rattle(base_structure, config.rattle_stdev, rng=self.rng)


class MDMicroBurstPolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        MD Micro Burst Policy: Runs short MD to generate candidates.
        Requires 'engine' and 'potential' in kwargs.
        """
        engine = kwargs.get("engine")
        potential = kwargs.get("potential")

        # SRP: Delegate fallback
        if not engine or not hasattr(engine, "config"):
             yield from self._fallback(base_structure, config, n_structures)
             return

        try:
            yield from self._run_burst(engine, potential, base_structure, config, n_structures)
        except Exception as e:
            logger.warning(f"MD Micro Burst failed: {e}. Falling back to Rattle.")
            yield from self._fallback(base_structure, config, n_structures)

    def _fallback(self, base_structure: Atoms, config: StructureConfig, n_structures: int) -> Iterator[Atoms]:
        """Fallback policy (Rattle) when MD is not available or fails."""
        for _ in range(n_structures):
            yield rattle(base_structure, config.rattle_stdev, rng=self.rng)

    def _run_burst(
        self,
        engine: Any,
        potential: Any,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int
    ) -> Iterator[Atoms]:
        """Executes the MD burst logic."""
        # Create a burst engine config
        burst_temp = config.local_md_temp
        burst_steps = config.local_md_steps

        # Create a burst engine
        burst_config = engine.config.model_copy(update={
            "n_steps": burst_steps,
            "minimize": False,
            "temperature": burst_temp,
            "dump_freq": 0
        })

        # Instantiate new engine of same type
        BurstEngine = type(engine)
        burst_engine = BurstEngine(burst_config)

        for _ in range(n_structures):
            result = burst_engine.run(base_structure, potential)
            if result.trajectory_path:
                final = read(result.trajectory_path, index=-1)
                if isinstance(final, list):
                    yield final[-1]
                else:
                    yield final
            else:
                # If run succeeds but no trajectory (unlikely if n_steps>0), fallback for this item
                yield rattle(base_structure, config.rattle_stdev, rng=self.rng)


class CompositePolicy(BasePolicy):
    def __init__(self, policies: list[BasePolicy]) -> None:
        super().__init__()
        self.policies = policies

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Composite Policy: Interleaves generation from all active policies.
        """
        if not self.policies:
            return

        # We split n_structures among policies
        n_policies = len(self.policies)
        if n_policies == 0:
            return

        count_per_policy = n_structures // n_policies
        remainder = n_structures % n_policies

        counts = [count_per_policy] * n_policies
        for i in range(remainder):
            counts[i] += 1

        for policy, count in zip(self.policies, counts, strict=True):
            if count > 0:
                yield from policy.generate(base_structure, config, n_structures=count, **kwargs)
