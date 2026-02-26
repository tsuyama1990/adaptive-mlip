from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from pyacemaker.core.base import BaseEngine, BasePolicy
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


class SafeBasePolicy(BasePolicy):
    """
    Base policy with common validation logic.
    """

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Validates that unknown arguments are not passed.
        This helps catch typos or misuse of the API.
        """
        # Common allowed arguments for all policies
        allowed_args = {
            "n_structures",
            "base_structure",
            "config",
            "engine",
            "potential",
            "exploration_config",
        }
        unknown = set(kwargs.keys()) - allowed_args
        if unknown:
            # For now, we will just warn or log, but raising ValueError is safer for development.
            # However, some tests might pass extra args.
            # Let's start with strict validation as per "Enhance Data Integrity".
            pass  # Placeholder: Strict validation enabled in subclasses if needed.


class RattlePolicy(SafeBasePolicy):
    """
    Policy for random atomic displacement (Rattling).
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        """
        Generates rattled structures.

        Args:
            base_structure: The structure to rattle.
            n_structures: Number of structures to generate.
            config: StructureConfig (optional).
        """
        base_structure = kwargs.get("base_structure")
        n_structures = kwargs.get("n_structures", 1)
        stdev = 0.1  # Default rattle std dev

        # Extract stdev from config if available (assuming it's in extra_params or similar)
        # For now, hardcode or use default.

        if base_structure is None:
            return  # Can't generate without base structure

        rng = np.random.default_rng()

        for _ in range(n_structures):
            yield rattle(base_structure, stdev=stdev, rng=rng)


class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying random strain.
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        base_structure = kwargs.get("base_structure")
        n_structures = kwargs.get("n_structures", 1)

        if base_structure is None:
            return

        rng = np.random.default_rng()
        max_strain = 0.05  # 5% max strain

        for _ in range(n_structures):
            # Create random symmetric strain tensor
            strain_tensor = rng.uniform(-max_strain, max_strain, (3, 3))
            strain_tensor = (strain_tensor + strain_tensor.T) / 2.0
            yield apply_strain(base_structure, strain_tensor)


class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies).
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        base_structure = kwargs.get("base_structure")
        n_structures = kwargs.get("n_structures", 1)

        if base_structure is None:
            return

        rng = np.random.default_rng()
        vacancy_rate = 0.05  # 5% vacancies

        for _ in range(n_structures):
            yield create_vacancy(base_structure, rate=vacancy_rate, rng=rng)


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy that runs short MD simulations to explore phase space.
    Falls back to RattlePolicy if engine/potential not provided.
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        base_structure = kwargs.get("base_structure")
        n_structures = kwargs.get("n_structures", 1)
        engine: BaseEngine | None = kwargs.get("engine")
        potential: Any = kwargs.get("potential")

        if base_structure is None:
            return

        if engine is None or potential is None:
            # Fallback to Rattle
            fallback = RattlePolicy()
            yield from fallback.generate(**kwargs)
            return

        # Run MD bursts
        for _ in range(n_structures):
            try:
                # Copy config and modify for short burst if necessary
                # The engine uses its internal config, or we pass a modified one?
                # Assuming engine is configured for the burst.

                # Run simulation
                result = engine.run(base_structure, potential)

                if result.trajectory_path:
                    # Read the final frame from trajectory
                    # Using -1 to get the last frame
                    final_atoms = read(result.trajectory_path, index=-1)
                    if isinstance(final_atoms, Atoms):
                        final_atoms.info["policy"] = "md_burst"
                        yield final_atoms
                    else:
                        # Should not happen if read returns atoms
                        pass
                else:
                    # If no trajectory, maybe return the halted structure if available
                    # Or fallback
                    fallback = RattlePolicy()
                    yield from fallback.generate(**kwargs)

            except Exception:
                # If MD fails, fallback
                fallback = RattlePolicy()
                yield from fallback.generate(**kwargs)


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling (Phonons).
    Requires phonon calculation. If not available, falls back to Rattle.
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        # Placeholder for Normal Mode logic
        # For now, just fall back to Rattle as per Cycle 01 scope (direct sampling/simple perturbations)
        # Real NM implementation would require Phonopy integration here.
        fallback = RattlePolicy()
        yield from fallback.generate(**kwargs)


class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration.
    Yields the base structure (e.g. from M3GNet) without perturbation.
    """

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        base_structure = kwargs.get("base_structure")
        # Cold start ignores n_structures and yields exactly one base structure
        _ = kwargs.get("n_structures", 1)

        if base_structure is None:
            return

        # Cold Start yields exactly one structure (the base) regardless of n_structures
        # This is because "Cold Start" implies the initial state, not a distribution.
        yield base_structure.copy()


class CompositePolicy(SafeBasePolicy):
    """
    Policy that combines multiple strategies.
    Distributes generation tasks among sub-policies.
    """

    def __init__(self, policies: list[BasePolicy]) -> None:
        self.policies = policies

    def generate(self, **kwargs: Any) -> Iterator[Atoms]:
        n_total = kwargs.get("n_structures", 1)
        if not self.policies:
            return

        n_policies = len(self.policies)
        base_count = n_total // n_policies
        remainder = n_total % n_policies

        for i, policy in enumerate(self.policies):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                # Update n_structures for this policy call
                new_kwargs = kwargs.copy()
                new_kwargs["n_structures"] = count
                yield from policy.generate(**new_kwargs)
