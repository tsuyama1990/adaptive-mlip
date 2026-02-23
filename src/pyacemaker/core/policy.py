from abc import ABC, abstractmethod

from ase import Atoms

from pyacemaker.core.m3gnet_wrapper import M3GNetWrapper
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.utils.perturbations import apply_strain, create_vacancy, rattle


class BasePolicy(ABC):
    """Abstract base class for exploration policies."""

    def __init__(self, config: StructureConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, structure: Atoms | None = None) -> Atoms:
        """
        Applies the policy to the given structure.

        Args:
            structure: The base structure to modify (optional for generation policies).

        Returns:
            A new Atoms object.
        """


class ColdStartPolicy(BasePolicy):
    """Generates initial structures using M3GNet (or mock)."""

    def __init__(self, config: StructureConfig) -> None:
        super().__init__(config)
        self.m3gnet = M3GNetWrapper()

    def apply(self, structure: Atoms | None = None) -> Atoms:
        # Structure is ignored as we generate from scratch based on composition
        composition = "".join(self.config.elements)
        return self.m3gnet.predict_structure(composition)


class RattlePolicy(BasePolicy):
    """Perturbs atomic positions with Gaussian noise."""

    def apply(self, structure: Atoms | None = None) -> Atoms:
        if structure is None:
            msg = "RattlePolicy requires an input structure."
            raise ValueError(msg)
        return rattle(structure, self.config.rattle_stdev)


class StrainPolicy(BasePolicy):
    """Applies random strain to the unit cell."""

    def apply(self, structure: Atoms | None = None) -> Atoms:
        if structure is None:
            msg = "StrainPolicy requires an input structure."
            raise ValueError(msg)
        return apply_strain(structure, self.config.strain_range, self.config.strain_mode)


class DefectPolicy(BasePolicy):
    """Creates vacancies in the structure."""

    def apply(self, structure: Atoms | None = None) -> Atoms:
        if structure is None:
            msg = "DefectPolicy requires an input structure."
            raise ValueError(msg)
        return create_vacancy(structure, self.config.vacancy_rate)
