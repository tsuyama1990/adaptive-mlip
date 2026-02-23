from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms


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


class BaseOracle(ABC):
    """
    Abstract base class for property calculation (Oracle).
    Implementations typically wrap DFT codes like Quantum Espresso or VASP.
    """

    @abstractmethod
    def compute(self, structures: Iterator[Atoms], batch_size: int = 10) -> Iterator[Atoms]:
        """
        Computes properties (energy, forces, stress) for the given structures.

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
    def train(self, training_data_path: str | Path) -> Any:
        """
        Trains a potential using the provided training data file.

        To ensure scalability, training data should be passed as a file path
        rather than an in-memory list.

        Args:
            training_data_path: Path to the file containing labelled structures (e.g., .xyz, .pckl).

        Returns:
            Trained potential object or path to potential file.

        Raises:
            RuntimeError: If training fails (e.g., MLIP code crash, insufficient data).
            FileNotFoundError: If training data file does not exist.

        Example:
            class PacemakerTrainer(BaseTrainer):
                def train(self, path):
                    subprocess.run(["pace_train", "--dataset", str(path)])
                    return "potential.yace"
        """


class BaseEngine(ABC):
    """
    Abstract base class for simulation engine (MD/MC).
    Implementations wrap codes like LAMMPS or EON.
    """

    @abstractmethod
    def run(self, structure: Atoms | None, potential: Any) -> Any:
        """
        Runs a simulation using the given structure and potential.

        Args:
            structure: Initial structure. May be None if engine loads from file/config.
            potential: Trained potential. May be None if engine loads from file/config.

        Returns:
            Simulation result (trajectory, final structure, etc.).

        Raises:
            RuntimeError: If simulation fails (e.g., segmentation fault, physics explosion).

        Example:
            class LAMMPSEngine(BaseEngine):
                def run(self, structure, potential):
                    write_lammps_input(structure, potential)
                    subprocess.run(["lmp", ...])
                    return read_trajectory("dump.lammpstrj")
        """
