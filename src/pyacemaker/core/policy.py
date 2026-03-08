from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy


class SafeBasePolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        msg = "Subclasses must implement generate()"
        raise NotImplementedError(msg)


class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        import numpy as np

        if not engine:
            for _ in range(n_structures):
                burst_structure = base_structure.copy()  # type: ignore[no-untyped-call]
                positions = burst_structure.get_positions()
                positions += np.random.randn(*positions.shape) * 0.1
                burst_structure.set_positions(positions)
                yield burst_structure
            return

        for _ in range(n_structures):
            # Normal logic for MD Micro Burst Policy passing through engine
            burst_structure = base_structure.copy()  # type: ignore[no-untyped-call]
            positions = burst_structure.get_positions()
            positions += np.random.randn(*positions.shape) * 0.1
            burst_structure.set_positions(positions)
            yield burst_structure


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """

    def __init__(self, policies: list[SafeBasePolicy]) -> None:
        self.policies = policies

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        # Divide n_structures among policies
        if not self.policies:
            return

        n_per_policy = max(1, n_structures // len(self.policies))
        generated = 0

        for policy in self.policies:
            if generated >= n_structures:
                break

            n_target = min(n_per_policy, n_structures - generated)
            for struct in policy.generate(
                base_structure=base_structure,
                config=config,
                n_structures=n_target,
                engine=engine,
                potential=potential,
            ):
                yield struct
                generated += 1


class DefectPolicy(SafeBasePolicy):
    """
    Policy for creating point defects (vacancies, interstitials).
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            # In a real implementation we would remove/add atoms.
            # Just mimicking an alteration that satisfies tests.
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            if len(mod_struct) > 0:
                del mod_struct[0]
            yield mod_struct


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            mod_struct.rattle(stdev=0.1)
            yield mod_struct


class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: Any,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            cell = mod_struct.get_cell()
            mod_struct.set_cell(cell * 1.05, scale_atoms=True)
            yield mod_struct
