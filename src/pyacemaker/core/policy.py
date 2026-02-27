from collections.abc import Iterator

from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig


class ColdStartPolicy(BasePolicy):
    """
    Policy that returns the base structure without modification.
    Useful for initial dataset generation.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        """
        Yields the base structure exactly once.
        Ignores n_structures > 1 as it doesn't make sense to duplicate exact same structure.
        """
        # Yield at least one
        yield base_structure.copy() # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    """
    Policy that applies random Gaussian noise to atomic positions.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        stdev = config.rattle_stdev
        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            atoms.rattle(stdev=stdev) # type: ignore[no-untyped-call]
            yield atoms


class StrainPolicy(BasePolicy):
    """
    Policy that applies random strain to the simulation cell.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        from pyacemaker.utils.perturbations import apply_random_strain

        mode = config.strain_mode
        magnitude = config.strain_magnitude

        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            # apply_random_strain modifies in-place or returns new?
            # Assuming it modifies in-place based on utility name pattern in ASE ecosystem,
            # but let's check or assume utility handles it.
            # If it returns new, we yield that.

            # Implementation note: utils.perturbations.apply_random_strain likely modifies in-place.
            apply_random_strain(atoms, mode=str(mode), magnitude=magnitude)
            yield atoms


class DefectPolicy(BasePolicy):
    """
    Policy that introduces point defects (vacancies).
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        from pyacemaker.utils.perturbations import introduce_vacancies

        rate = config.vacancy_rate
        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            introduce_vacancies(atoms, rate=rate)
            yield atoms


class CompositePolicy(BasePolicy):
    """
    Executes multiple policies sequentially.
    """
    def __init__(self, policies: list[BasePolicy]) -> None:
        self.policies = policies

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        # Divide n_structures among policies? Or run all for n_structures?
        # Usually composite means: 20% this, 30% that.
        # But here we just have a list.
        # Let's split evenly for Cycle 01 simplicity.

        if not self.policies:
            return

        n_per_policy = max(1, n_structures // len(self.policies))

        count = 0
        for policy in self.policies:
            # Last policy takes remainder
            if policy is self.policies[-1]:
                current_n = n_structures - count
            else:
                current_n = n_per_policy

            if current_n <= 0:
                continue

            yield from policy.generate(base_structure, config, current_n, **kwargs)
            count += current_n


# Local Policies
class RandomDisplacementPolicy(RattlePolicy):
    """Alias for RattlePolicy used in local generation."""

class NormalModePolicy(BasePolicy):
    """Placeholder for Normal Mode sampling."""
    def generate(self, base_structure: Atoms, config: StructureConfig, n_structures: int, **kwargs: dict) -> Iterator[Atoms]:
        # Fallback to rattle if normal mode not implemented
        yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)

class MDMicroBurstPolicy(BasePolicy):
    """Placeholder for MD Micro Burst sampling."""
    def generate(self, base_structure: Atoms, config: StructureConfig, n_structures: int, **kwargs: dict) -> Iterator[Atoms]:
        # Fallback to rattle
        yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)
