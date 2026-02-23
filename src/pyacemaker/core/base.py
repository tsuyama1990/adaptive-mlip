from abc import ABC, abstractmethod
from typing import Any

from ase import Atoms


class BaseGenerator(ABC):
    """Abstract base class for structure generation."""

    @abstractmethod
    def generate(self, n_candidates: int) -> list[Atoms]:
        """
        Generates candidate structures.

        Args:
            n_candidates: Number of structures to generate.

        Returns:
            List of generated ASE Atoms objects.
        """


class BaseOracle(ABC):
    """Abstract base class for property calculation (Oracle)."""

    @abstractmethod
    def compute(self, structures: list[Atoms]) -> list[Atoms]:
        """
        Computes properties (energy, forces, stress) for the given structures.

        Args:
            structures: List of ASE Atoms objects.

        Returns:
            List of ASE Atoms objects with computed properties attached.
        """


class BaseTrainer(ABC):
    """Abstract base class for potential training."""

    @abstractmethod
    def train(self, training_data: list[Atoms]) -> Any:
        """
        Trains a potential using the provided training data.

        Args:
            training_data: List of labelled ASE Atoms objects.

        Returns:
            Trained potential object or path to potential file.
        """


class BaseEngine(ABC):
    """Abstract base class for simulation engine (MD/MC)."""

    @abstractmethod
    def run(self, structure: Atoms, potential: Any) -> Any:
        """
        Runs a simulation using the given structure and potential.

        Args:
            structure: Initial structure.
            potential: Trained potential.

        Returns:
            Simulation result (trajectory, final structure, etc.).
        """
