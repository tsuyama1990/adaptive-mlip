from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.domain_models.data import AtomStructure
from pyacemaker.domain_models.md import MDSimulationResult


class BasePolicy(ABC):
    """
    Abstract base class for exploration policies.
    """
    @abstractmethod
    def generate(self, **kwargs: Any) -> None:
        """
        Generates new candidates based on policy logic.
        """


class BaseGenerator(ABC):
    """
    Abstract base class for structure generation.
    Implementations should explore chemical space to create candidate structures.
    """

    @abstractmethod
    def update_config(self, config: Any) -> None:
        """
        Updates the generator configuration.
        This allows adaptive policies to modify generation parameters at runtime.

        Args:
            config: New configuration object (e.g. StructureConfig).
        """

    @abstractmethod
    def generate(self, n_candidates: int) -> Iterator[AtomStructure]:
        """
        Generates candidate structures.

        This method should return an iterator to allow streaming of large datasets
        without loading everything into memory.

        Args:
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding AtomStructure objects.
            If generation cannot produce any structures, the iterator should be empty
            (or raise an error if 0 is invalid for the context).

        Raises:
            RuntimeError: If generation fails due to internal errors or configuration issues.
            ValueError: If input parameters are invalid.
        """

    @abstractmethod
    def generate_local(self, base_structure: Atoms, n_candidates: int, **kwargs: Any) -> Iterator[AtomStructure]:
        """
        Generates candidate structures by perturbing a base structure.
        Used in OTF loops to explore the local neighborhood of a high-uncertainty configuration.

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.
            **kwargs: Additional arguments (e.g., engine).

        Returns:
            Iterator yielding AtomStructure objects.
        """


class BaseOracle(ABC):
    """
    Abstract base class for property calculation (Oracle).
    Implementations typically wrap DFT codes like Quantum Espresso or VASP.
    """

    @abstractmethod
    def compute(self, structures: Iterator[AtomStructure], batch_size: int = 10) -> Iterator[AtomStructure]:
        """
        Computes properties (energy, forces, stress) for the given structures.

        Args:
            structures: Iterator of AtomStructure objects.
            batch_size: Number of structures to compute in a single batch (if supported).

        Returns:
            Iterator of AtomStructure objects with computed properties attached (e.g. in atoms.info).
            If the input iterator is empty, the returned iterator should also be empty.

        Raises:
            RuntimeError: If calculation fails (e.g., DFT convergence error, connection error).
            ValueError: If input structures are invalid.
        """


class BaseTrainer(ABC):
    """
    Abstract base class for potential training.
    Implementations wrap MLIP codes like Pacemaker, NequIP, or MACE.
    """

    @abstractmethod
    def train(
        self,
        training_data_path: str | Path,
        initial_potential: str | Path | None = None
    ) -> Any:
        """
        Trains a potential using the provided training data file.

        To ensure scalability, training data should be passed as a file path
        rather than an in-memory list.

        Args:
            training_data_path: Path to the file containing labelled structures (e.g., .xyz, .pckl).
            initial_potential: Optional path to an existing potential to fine-tune from.

        Returns:
            Trained potential object or path to potential file.

        Raises:
            RuntimeError: If training fails (e.g., MLIP code crash, insufficient data).
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
            structure: Initial structure. May be None if engine loads from file/config.
            potential: Trained potential. May be None if engine loads from file/config.

        Returns:
            MDSimulationResult containing trajectory path, halt status, etc.

        Raises:
            RuntimeError: If simulation fails (e.g., segmentation fault, physics explosion).
        """

    @abstractmethod
    def compute_static_properties(self, structure: Atoms, potential: Any) -> MDSimulationResult:
        """
        Computes static properties (energy, forces, stress) for a structure.
        Equivalent to a 0-step MD run or minimization.

        Args:
            structure: Structure to compute properties for.
            potential: Potential to use.

        Returns:
            MDSimulationResult containing energy, forces, etc.
        """

    @abstractmethod
    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        """
        Relaxes the structure to a local minimum.

        Args:
            structure: Structure to relax.
            potential: Potential to use.

        Returns:
            Relaxed structure as an ASE Atoms object.

        Raises:
            RuntimeError: If relaxation fails.
        """
