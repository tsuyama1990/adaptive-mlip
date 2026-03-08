import copy
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from pyacemaker.core.base import BaseEngine, BasePolicy
from pyacemaker.domain_models.constants import (
    POLICY_MICROBURST_N_STEPS,
    POLICY_MICROBURST_NOISE_STDEV,
    POLICY_NORMALMODE_NOISE_STDEV,
)
from pyacemaker.domain_models.structure import StructureConfig


class SafeBasePolicy(BasePolicy):
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
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
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            yield base_structure.copy()  # type: ignore[no-untyped-call]


class MDMicroBurstPolicy(SafeBasePolicy):
    """
    Policy using short MD bursts to explore phase space.
    """

    def generate(  # noqa: C901
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        if not engine:
            for _ in range(n_structures):
                burst_structure = base_structure.copy()  # type: ignore[no-untyped-call]
                # It is safer in ase to retrieve positions, manipulate, and set them back
                positions = burst_structure.get_positions()
                positions += np.random.randn(*positions.shape) * POLICY_MICROBURST_NOISE_STDEV
                burst_structure.set_positions(positions)
                yield burst_structure
            return

        for _ in range(n_structures):
            # Using MD Engine for actual burst exploration
            # Ensure safe config override for microburst
            if hasattr(engine, "config"):
                burst_config = copy.deepcopy(engine.config)
                if hasattr(burst_config, "n_steps"):
                    burst_config.n_steps = POLICY_MICROBURST_N_STEPS  # Micro burst short steps

            if isinstance(engine, BaseEngine):
                result = engine.run(
                    structure=base_structure, potential=potential
                )

                # In a real implementation we would load result.trajectory_path
                # But for architecture completeness, yield rattled structure or loaded structure
                if result and result.trajectory_path:
                    traj_path = Path(result.trajectory_path)
                    if traj_path.exists() and traj_path.is_file():
                        try:
                            traj = read(traj_path, index=":")
                            if traj and isinstance(traj, list):
                                atoms = traj[-1]
                                if isinstance(atoms, Atoms):
                                    yield atoms
                                    continue
                        except OSError:
                            # specifically handle file read errors
                            pass

            # Fallback if engine run doesn't produce loaded structure
            burst_structure = base_structure.copy()  # type: ignore[no-untyped-call]
            positions = burst_structure.get_positions()
            positions += np.random.randn(*positions.shape) * POLICY_MICROBURST_NOISE_STDEV
            burst_structure.set_positions(positions)
            yield burst_structure


class NormalModePolicy(SafeBasePolicy):
    """
    Policy using Normal Mode sampling.
    """

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            positions = mod_struct.get_positions()

            # Placeholder for actual normal mode displacement
            displacement = np.random.randn(*positions.shape) * POLICY_NORMALMODE_NOISE_STDEV
            positions += displacement

            mod_struct.set_positions(positions)

            yield mod_struct


class CompositePolicy(SafeBasePolicy):
    """
    Composite Policy that can combine multiple exploration strategies.
    """

    def __init__(self, policies: list[SafeBasePolicy]) -> None:
        self.policies = policies

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
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
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            if len(mod_struct) > 0:
                idx_to_remove = secrets.randbelow(len(mod_struct))
                del mod_struct[idx_to_remove]
            yield mod_struct


class RattlePolicy(SafeBasePolicy):
    """
    Policy for rattling structures (random perturbation).
    """

    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
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
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        for _ in range(n_structures):
            mod_struct = base_structure.copy()  # type: ignore[no-untyped-call]
            cell = mod_struct.get_cell()
            mod_struct.set_cell(cell * 1.05, scale_atoms=True)
            yield mod_struct
