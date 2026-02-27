from collections.abc import Iterator

from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig


class ColdStartPolicy(BasePolicy):
    """
    Policy that returns the base structure without modification.
    Useful for initial dataset generation (Exploration Phase 0).
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

        Args:
            base_structure: The template structure (e.g., from M3GNet).
            config: Structure generation configuration.
            n_structures: Requested number of structures (ignored, always 1).
            **kwargs: Additional context.

        Yields:
            Atoms: A copy of the base structure.
        """
        # Yield at least one
        yield base_structure.copy() # type: ignore[no-untyped-call]


class RattlePolicy(BasePolicy):
    """
    Policy that applies random Gaussian noise to atomic positions.
    Explores local potential energy surface near equilibrium.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        """
        Generates n_structures by randomly rattling the base structure.

        Args:
            base_structure: The template structure.
            config: Structure generation configuration (uses rattle_stdev).
            n_structures: Number of structures to generate.
            **kwargs: Additional context.

        Yields:
            Atoms: Rattled structures.
        """
        stdev = config.rattle_stdev
        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            atoms.rattle(stdev=stdev) # type: ignore[no-untyped-call]
            yield atoms


class StrainPolicy(BasePolicy):
    """
    Policy that applies random strain to the simulation cell.
    Explores cell volume and shape variations.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        """
        Generates n_structures by applying random strain.

        Args:
            base_structure: The template structure.
            config: Structure generation configuration (uses strain_mode, strain_magnitude).
            n_structures: Number of structures to generate.
            **kwargs: Additional context.

        Yields:
            Atoms: Strained structures.
        """
        from pyacemaker.utils.perturbations import apply_random_strain

        mode = config.strain_mode
        magnitude = config.strain_magnitude

        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            # apply_random_strain modifies in-place
            apply_random_strain(atoms, mode=str(mode), magnitude=magnitude)
            yield atoms


class DefectPolicy(BasePolicy):
    """
    Policy that introduces point defects (vacancies).
    Explores off-stoichiometry configurations.
    """
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int,
        **kwargs: dict,
    ) -> Iterator[Atoms]:
        """
        Generates n_structures with random vacancies.

        Args:
            base_structure: The template structure.
            config: Structure generation configuration (uses vacancy_rate).
            n_structures: Number of structures to generate.
            **kwargs: Additional context.

        Yields:
            Atoms: Structures with vacancies.
        """
        from pyacemaker.utils.perturbations import introduce_vacancies

        rate = config.vacancy_rate
        for _ in range(n_structures):
            atoms = base_structure.copy() # type: ignore[no-untyped-call]
            introduce_vacancies(atoms, rate=rate)
            yield atoms


class CompositePolicy(BasePolicy):
    """
    Executes multiple policies sequentially.
    Allows mixing different exploration strategies (e.g., 50% rattle, 50% strain).
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
        """
        Generates structures by delegating to sub-policies.
        Currently splits n_structures evenly among active policies.

        Args:
            base_structure: The template structure.
            config: Structure generation configuration.
            n_structures: Total number of structures to generate.
            **kwargs: Additional context.

        Yields:
            Atoms: Generated structures from all sub-policies.
        """
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
    """Alias for RattlePolicy used in local generation strategy."""

class NormalModePolicy(BasePolicy):
    """
    Placeholder for Normal Mode sampling.
    Future implementation will use phonon modes for perturbations.
    """
    def generate(self, base_structure: Atoms, config: StructureConfig, n_structures: int, **kwargs: dict) -> Iterator[Atoms]:
        """Fallback to rattle if normal mode not implemented."""
        yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)

class MDMicroBurstPolicy(BasePolicy):
    """
    Placeholder for MD Micro Burst sampling.
    Future implementation will run short MD trajectories.
    """
    def generate(self, base_structure: Atoms, config: StructureConfig, n_structures: int, **kwargs: dict) -> Iterator[Atoms]:
        """Fallback to rattle if MD burst not implemented."""
        yield from RattlePolicy().generate(base_structure, config, n_structures, **kwargs)
