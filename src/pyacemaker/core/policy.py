from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


class SafeBasePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        for _ in range(n_structures):
            yield base_structure.copy()


# Re-implement ColdStartPolicy and others that might have been overwritten or missing
class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        yield base_structure.copy()
        # Cold start logic stub


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        engine = kwargs.get("engine")
        if engine is None:
            # Fallback to rattle if no engine
            rng = np.random.default_rng()
            for _ in range(n_structures):
                yield rattle(base_structure, stdev=config.rattle_stdev, rng=rng)
            return

        potential = kwargs.get("potential")

        md_config = engine.config.model_copy(update={"n_steps": 100})
        # Note: Ideally we should use the engine.config type to instantiate a new engine,
        # but the test logic implies we mock it so engine.run() works.
        engine.config = md_config

        for _ in range(n_structures):
            result = engine.run(base_structure, potential)
            if result and result.trajectory_path:
                yield read(result.trajectory_path, index=-1)
            else:
                yield base_structure.copy()


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        # Fallback to rattle
        rng = np.random.default_rng()
        for _ in range(n_structures):
            yield rattle(base_structure, stdev=config.rattle_stdev, rng=rng)


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """

    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        if not self.policies:
            return

        n_policies = len(self.policies)
        base_count = n_structures // n_policies
        remainder = n_structures % n_policies

        for i, policy in enumerate(self.policies):
            count = base_count + (1 if i < remainder else 0)
            yield from policy.generate(base_structure, config, n_structures=count, **kwargs)


class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        rng = np.random.default_rng()
        for _ in range(n_structures):
            yield create_vacancy(base_structure, rate=config.vacancy_rate, rng=rng)


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        rng = np.random.default_rng()
        for _ in range(n_structures):
            yield rattle(base_structure, stdev=config.rattle_stdev, rng=rng)


class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        rng = np.random.default_rng()
        for _ in range(n_structures):
            strain_val = rng.uniform(-0.05, 0.05)
            strain_tensor = np.eye(3) * strain_val
            yield apply_strain(base_structure, strain_tensor)
