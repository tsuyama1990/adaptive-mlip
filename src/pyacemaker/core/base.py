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
    def generate(self, n_candidates: int) -> Iterator[Atoms]:
        """
        Generates candidate structures.

        This method should return an iterator to allow streaming of large datasets
        without loading everything into memory.

        Args:
            n_candidates: Number of structures to generate.

        Returns:
            Iterator yielding ASE Atoms objects.

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

        Example:
            class LAMMPSEngine(BaseEngine):
                def run(self, structure, potential):
                    write_lammps_input(structure, potential)
                    subprocess.run(["lmp", ...])
                    return read_trajectory("dump.lammpstrj")
        """
