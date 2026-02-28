from collections.abc import Iterable
from typing import Any

from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.domain_models.structure import StructureConfig


class SafeBasePolicy(BasePolicy):
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
        yield base_structure.copy()  # type: ignore[no-untyped-call]


class ColdStartPolicy(SafeBasePolicy):
    """
    Policy for initial exploration (Cold Start).
    Usually implies random structure generation or grid search.
    """


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
        from ase.io import read

        engine = kwargs.get("engine")
        potential = kwargs.get("potential")

        if not engine:
            for _ in range(n_structures):
                yield base_structure.copy()  # type: ignore[no-untyped-call]
            return

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            try:
                # We expect engine.run to return an MDSimulationResult
                result = engine.run(atoms, potential)
                target_path = result.halt_structure_path or result.trajectory_path
                if target_path:
                    # Trajectory might contain multiple frames, we want the last one
                    trajectory = read(target_path, index=":")
                    if trajectory and isinstance(trajectory, list):
                        yield trajectory[-1]
                    elif isinstance(trajectory, Atoms):
                        yield trajectory
                    else:
                        yield atoms
                else:
                    yield atoms
            except Exception:
                yield atoms


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
        import numpy as np

        stdev = getattr(config, "rattle_stdev", 0.1)
        masses = base_structure.get_masses()

        # If masses are zero (dummy atoms) or not set, fallback to 1.0 to avoid division by zero
        masses[masses <= 0] = 1.0

        # Scale displacements by inverse square root of mass
        scale_factors = 1.0 / np.sqrt(masses)
        # Normalize scale factors so the average displacement matches stdev
        scale_factors /= np.mean(scale_factors)

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            positions = atoms.get_positions()

            # Generate random displacements
            displacements = np.random.normal(scale=stdev, size=positions.shape)

            # Apply mass scaling
            displacements *= scale_factors[:, np.newaxis]

            positions += displacements
            atoms.set_positions(positions)
            yield atoms


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """

    def __init__(self, policies: list[BasePolicy] | None = None) -> None:
        self.policies = policies or []

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
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
    ) -> Iterable[Atoms]:
        import random

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            if len(atoms) > 0:
                idx = random.randint(0, len(atoms) - 1)
                del atoms[idx]
            yield atoms


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
        import numpy as np

        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            stdev = getattr(config, "rattle_stdev", 0.1)
            # Use random rattle directly instead of relying on the method which might be seeded
            # or do nothing on some setups in tests.
            positions = atoms.get_positions()
            positions += np.random.normal(scale=stdev, size=positions.shape)
            atoms.set_positions(positions)
            yield atoms


class StrainPolicy(SafeBasePolicy):
    """
    Policy for applying strain to structures.
    """

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterable[Atoms]:
        for _ in range(n_structures):
            atoms = base_structure.copy()  # type: ignore[no-untyped-call]
            cell = atoms.get_cell()
            strain_factor = getattr(config, "strain_factor", 1.05)
            atoms.set_cell(cell * strain_factor, scale_atoms=True)
            yield atoms
