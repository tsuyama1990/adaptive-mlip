from abc import ABC, abstractmethod
from collections.abc import Iterator
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
    def compute(self, structures: list[Atoms], batch_size: int = 10) -> list[Atoms]:
        """
        Computes properties (energy, forces, stress) for the given structures.

        Args:
            structures: List of ASE Atoms objects.
            batch_size: Number of structures to compute in a single batch (if supported).

        Returns:
            List of ASE Atoms objects with computed properties attached (e.g. in atoms.info).

        Example:
            class DFTOracle(BaseOracle):
                def compute(self, structures, batch_size=10):
                    for batch in chunk(structures, batch_size):
                        run_dft(batch)
                    return structures
        """


class BaseTrainer(ABC):
    """
    Abstract base class for potential training.
    Implementations wrap MLIP codes like Pacemaker, NequIP, or MACE.
    """

    @abstractmethod
    def train(self, training_data: list[Atoms]) -> Any:
        """
        Trains a potential using the provided training data.

        Args:
            training_data: List of labelled ASE Atoms objects.

        Returns:
            Trained potential object or path to potential file.

        Example:
            class PacemakerTrainer(BaseTrainer):
                def train(self, data):
                    write_data(data)
                    subprocess.run(["pace_train", ...])
                    return "potential.yace"
        """


class BaseEngine(ABC):
    """
    Abstract base class for simulation engine (MD/MC).
    Implementations wrap codes like LAMMPS or EON.
    """

    @abstractmethod
    def run(self, structure: Atoms, potential: Any) -> Any:
        """
        Runs a simulation using the given structure and potential.

        Args:
            structure: Initial structure.
            potential: Trained potential.

        Returns:
            Simulation result (trajectory, final structure, etc.).

        Example:
            class LAMMPSEngine(BaseEngine):
                def run(self, structure, potential):
                    write_lammps_input(structure, potential)
                    subprocess.run(["lmp", ...])
                    return read_trajectory("dump.lammpstrj")
        """
