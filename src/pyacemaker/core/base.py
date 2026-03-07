from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.domain_models.md import MDSimulationResult
from pyacemaker.domain_models.structure import StructureConfig


class BasePolicy(ABC):
    """
    Abstract base class for exploration policies.
    """

    @abstractmethod
    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """

    @abstractmethod
    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates candidates for local neighborhood exploration.
        """


class BaseGenerator(ABC):
    """
    Abstract base class for structure generation.
    Implementations should explore chemical space to create candidate structures.
    """

    @abstractmethod
    def update_config(self, config: StructureConfig) -> None:
        """
        Updates the generator configuration.
        This allows adaptive policies to modify generation parameters at runtime.

        Args:
            config: New configuration object.
        """
        StructureConfig.model_validate(config)

    @abstractmethod
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures.

        This method should return an iterator to allow streaming of large datasets
        without loading everything into memory.

        Args:
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding ASE Atoms objects.
            If generation cannot produce any structures, the iterator should be empty
            (or raise an error if 0 is invalid for the context).

        Raises:
            RuntimeError: If generation fails due to internal errors or configuration issues.
            ValueError: If input parameters are invalid.

        Example:
            class RandomGenerator(BaseGenerator):
                def generate(self, n):
                    for _ in range(n):
                        yield create_random_structure()
        """

    @abstractmethod
    def generate_local(
        self, base_structure: Atoms, n_candidates: int, **kwargs: Any
    ) -> Iterator[Atoms]:
        """
        Generates candidate structures by perturbing a base structure.
        Used in OTF loops to explore the local neighborhood of a high-uncertainty configuration.

        Args:
            base_structure: The reference structure to perturb.
            n_candidates: Number of structures to generate.
            **kwargs: Additional arguments (e.g., engine).

        Returns:
            Iterator yielding ASE Atoms objects.
        """


class BaseOracle(ABC):
    """
    Abstract base class for property calculation (Oracle).
    Implementations typically wrap DFT codes like Quantum Espresso or VASP.
    """

    @abstractmethod
    def compute_uncertainty(self, structure: Atoms) -> float:
        """
        Computes the uncertainty metric (e.g. gamma) for a single structure.
        """

    @abstractmethod
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties (energy, forces, stress) for the given structures.
        Must strictly accept an Iterator to guarantee O(1) memory profiling and should
        raise TypeError if a materialized list or tuple is provided.

        Args:
            structures: Iterator of ASE Atoms objects.
            batch_size: Number of structures to compute in a single batch (if supported).

        Returns:
            Iterator of ASE Atoms objects with computed properties attached (e.g. in atoms.info).
            If the input iterator is empty, the returned iterator should also be empty.

        Raises:
            RuntimeError: If calculation fails (e.g., DFT convergence error, connection error).
            ValueError: If input structures are invalid.

        Example:
            class DFTOracle(BaseOracle):
                def compute(self, structures, batch_size=10):
                    for batch in batched(structures, batch_size):
                        results = run_dft(batch)
                        for res in results:
                            yield res
        """


class BaseTrainer(ABC):
    """
    Abstract base class for potential training.
    Implementations wrap MLIP codes like Pacemaker, NequIP, or MACE.
    """

    @abstractmethod
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Path | None:
        """
        Trains a potential using the provided training data file.

        To ensure scalability, training data should be passed as a file path
        rather than an in-memory list. Implementations must validate these paths using
        `validate_path_safe()`.

        Args:
            training_data_path: Path to the file containing labelled structures (e.g., .xyz, .pckl).
            initial_potential: Optional path to an existing potential to fine-tune from.

        Returns:
            Trained potential object or path to potential file.

        Raises:
            RuntimeError: If training fails (e.g., MLIP code crash, insufficient data).
            FileNotFoundError: If training data file does not exist.
        """

    @abstractmethod
    def incremental_train(
        self, new_data_path: str | Path, replay_buffer_path: str | Path | None = None, replay_buffer_size: int = 500
    ) -> Path | None:
        """
        Performs Delta Learning using an active replay buffer to prevent catastrophic forgetting.
        """

    @abstractmethod
    def get_replay_buffer(self) -> Path | None:
        """
        Returns the path to the current replay buffer.
        """


class BaseEngine(ABC):
    """
    Abstract base class for simulation engine (MD/MC).
    Implementations wrap codes like LAMMPS or EON.
    """

    @abstractmethod
    def save_state(self, path: str | Path) -> None:
        """
        Saves the internal state of the engine.
        """

    @abstractmethod
    def load_state(self, path: str | Path) -> None:
        """
        Loads the internal state of the engine.
        """

    @abstractmethod
    def run(self, structure: Atoms | None, potential: Any, **kwargs: Any) -> MDSimulationResult:
        """
        Runs a simulation using the given structure and potential.

        Args:
            structure: Initial structure. May be None if engine loads from file/config.
            potential: Trained potential. May be None if engine loads from file/config.

        Returns:
            MDSimulationResult containing trajectory path, halt status, etc.

        Raises:
            RuntimeError: If simulation fails (e.g., segmentation fault, physics explosion).

        Example:
            class LAMMPSEngine(BaseEngine):
                def run(self, structure, potential):
                    write_lammps_input(structure, potential)
                    subprocess.run(["lmp", ...])
                    return MDSimulationResult(...)
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
