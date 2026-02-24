from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.domain_models.md import MDSimulationResult


class BaseGenerator(ABC):
    """
    Abstract base class for structure generation strategies.
    Implementations explore chemical space to create candidate structures.
    """

    @abstractmethod
    def update_config(self, config: Any) -> None:
        """
        Updates the generator configuration at runtime.

        Args:
            config: New configuration object (e.g. StructureConfig).
        """

    @abstractmethod
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures.

        This method MUST return an iterator to allow streaming of large datasets
        without loading everything into memory.

        Args:
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding ASE Atoms objects.

        Raises:
            GeneratorError: If generation fails due to internal errors.
            ValueError: If input parameters are invalid.
        """

    @abstractmethod
    def generate_local(self, base_structure: Atoms, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures by perturbing a base structure (local exploration).

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding ASE Atoms objects.
        """


class BaseOracle(ABC):
    """
    Abstract base class for property calculation (Oracle).
    Implementations typically wrap DFT codes (QE, VASP) or high-fidelity potentials.
    """

    @abstractmethod
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties (energy, forces, stress) for the given structures.

        Args:
            structures: Iterator of ASE Atoms objects.
            batch_size: Number of structures to compute in a single batch.

        Returns:
            Iterator of ASE Atoms objects with computed properties attached.

        Raises:
            OracleError: If calculation fails.
        """


class BaseTrainer(ABC):
    """
    Abstract base class for potential training.
    Implementations wrap MLIP codes like Pacemaker, NequIP, or MACE.
    """

    @abstractmethod
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        """
        Trains a potential using the provided training data file.

        Args:
            training_data_path: Path to the file containing labelled structures.
            initial_potential: Optional path to an existing potential to fine-tune from.

        Returns:
            Trained potential object or path to potential file.

        Raises:
            TrainerError: If training fails.
            FileNotFoundError: If training data file does not exist.
        """


class BaseEngine(ABC):
    """
    Abstract base class for simulation engine (MD/MC).
    Implementations wrap codes like LAMMPS or EON.
    """

    @abstractmethod
    def run(self, structure: Atoms | None, potential: Any) -> MDSimulationResult:
        """
        Runs a simulation using the given structure and potential.

        Args:
            structure: Initial structure.
            potential: Trained potential.

        Returns:
            MDSimulationResult containing trajectory path, halt status, etc.

        Raises:
            EngineError: If simulation fails.
        """
