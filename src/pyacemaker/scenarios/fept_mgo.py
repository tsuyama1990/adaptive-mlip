import contextlib
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from ase import Atom, Atoms
from ase.build import bulk, surface
from ase.io import write
from pydantic import BaseModel, Field, ValidationError

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.interfaces.eon_driver import EONWrapper
from pyacemaker.scenarios.base_scenario import BaseScenario

if TYPE_CHECKING:
    from pyacemaker.domain_models.config import PyAceConfig

logger = logging.getLogger(__name__)


class FePtMgoParameters(BaseModel):
    """Parameters for FePt on MgO scenario."""

    num_depositions: int = Field(10, ge=1, description="Number of atoms to deposit")
    fe_pt_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Ratio of Fe atoms (0.0 to 1.0)")
    potential_path: Path | None = Field(
        None, description="Path to potential file (overrides EON config)"
    )
    mgo_lattice_constant: float = Field(4.212, gt=0.0, description="Lattice constant of MgO")
    deposition_height: float = Field(
        2.5, gt=0.0, description="Height above surface for deposition (Angstroms)"
    )
    write_intermediate_files: bool = Field(
        True, description="Whether to write intermediate XYZ files"
    )
    max_retries: int = Field(
        3, ge=1, description="Maximum retries for relaxation during deposition"
    )
    random_seed: int | None = Field(None, description="Random seed for reproducibility")


class DepositionManager:
    """Manages the deposition of atoms onto a surface."""

    def __init__(self, engine: LammpsEngine, params: FePtMgoParameters, potential: Path) -> None:
        self.engine = engine
        self.params = params
        self.potential = potential

    def _select_element(self) -> str:
        """Selects element (Fe or Pt) based on ratio."""
        return "Fe" if random.random() < self.params.fe_pt_ratio else "Pt"  # noqa: S311

    def _calculate_deposition_position(self, structure: Atoms) -> Tuple[float, float, float]:
        """Calculates random x, y and max_z + height for deposition."""
        # Find max z of surface atoms
        if len(structure) > 0:
            # Use get_positions() to be safe across ASE versions
            positions = structure.get_positions()  # type: ignore[no-untyped-call]
            max_z = positions[:, 2].max()
        else:
            max_z = 0.0

        # Random x, y within cell
        lx = structure.cell[0, 0]
        ly = structure.cell[1, 1]
        x = random.uniform(0, lx)  # noqa: S311
        y = random.uniform(0, ly)  # noqa: S311
        z = max_z + self.params.deposition_height

        return x, y, z

    def _relax_structure_with_retries(self, structure: Atoms, step_index: int) -> Atoms | None:
        """Attempts to relax the structure with retries."""
        max_retries = self.params.max_retries

        for attempt in range(max_retries):
            try:
                # Relax returns a new Atoms object
                return self.engine.relax(structure, self.potential)
            except Exception as e:
                logger.warning(
                    "Deposition %d relaxation failed (attempt %d/%d): %s",
                    step_index,
                    attempt + 1,
                    max_retries,
                    e,
                )

        logger.error(
            "Failed to relax structure after deposition %d. Aborting deposition loop.", step_index
        )
        return None

    def deposit(self, slab: Atoms) -> Atoms:
        """Deposits atoms onto the slab."""
        num_depositions = self.params.num_depositions

        # Work on a copy initially
        structure = slab.copy()  # type: ignore[no-untyped-call]

        for i in range(num_depositions):
            element = self._select_element()
            x, y, z = self._calculate_deposition_position(structure)

            # Use Atom object to ensure position is set correctly
            atom = Atom(element, position=(x, y, z))  # type: ignore[no-untyped-call]
            structure.append(atom)

            # Relax structure using MD Engine with retries
            relaxed_structure = self._relax_structure_with_retries(structure, i)

            if relaxed_structure is not None:
                structure = relaxed_structure
            else:
                msg = f"Deposition failed at step {i} after {self.params.max_retries} attempts."
                raise RuntimeError(msg)

        return structure  # type: ignore[no-any-return]


class FePtMgoScenario(BaseScenario):
    """
    Implements the 'Grand Challenge': Fe/Pt deposition on MgO (001) surface
    followed by aKMC simulation for L10 ordering.
    """

    def __init__(
        self,
        config: "PyAceConfig",
        engine: LammpsEngine | None = None,
        eon_wrapper: EONWrapper | None = None,
    ) -> None:
        super().__init__(config)
        self.engine = engine or LammpsEngine(self.config.md)
        self.eon_wrapper = eon_wrapper

        # Validate parameters
        try:
            raw_params = self.config.scenario.parameters if self.config.scenario else {}
            self.params = FePtMgoParameters(**raw_params)
        except ValidationError as e:
            msg = f"Invalid parameters for FePtMgoScenario: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e

        # Initialize random seed if provided
        if self.params.random_seed is not None:
            random.seed(self.params.random_seed)

    def run(self) -> None:
        """Executes the full FePt/MgO workflow."""
        if not self.config.scenario or not self.config.scenario.enabled:
            msg = "Scenario configuration is missing or disabled."
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Starting FePt on MgO Scenario: %s", self.name)

        # 1. Generate MgO (001) Surface
        mgo_surface = self._generate_surface()
        if self.params.write_intermediate_files:
            try:
                write("mgo_surface.xyz", mgo_surface)
            except Exception as e:
                logger.warning("Failed to write mgo_surface.xyz: %s", e)
        logger.info("Generated MgO surface with %d atoms.", len(mgo_surface))

        # 2. Deposit Fe/Pt Atoms using DepositionManager
        potential = self._get_potential_path()
        deposition_manager = DepositionManager(self.engine, self.params, potential)

        deposited_structure = deposition_manager.deposit(mgo_surface)

        if self.params.write_intermediate_files:
            try:
                write("deposited.xyz", deposited_structure)
            except Exception as e:
                logger.warning("Failed to write deposited.xyz: %s", e)
        logger.info("Deposition complete. Total atoms: %d", len(deposited_structure))

        # 3. Run aKMC using EON
        if self.config.eon and self.config.eon.enabled:
            self._run_akmc(deposited_structure)
        else:
            logger.info("EON is disabled in config. Skipping aKMC step.")

    def _generate_surface(self) -> Atoms:
        """Generates MgO (001) surface."""
        a = self.params.mgo_lattice_constant
        mgo = bulk("MgO", "rocksalt", a=a)
        return surface(mgo, (0, 0, 1), layers=4, vacuum=10.0)  # type: ignore[no-untyped-call, no-any-return]

    def _get_potential_path(self) -> Path:
        """Resolves potential path from params or config."""
        potential = None
        # Prioritize params over EON config as per docstring
        if self.params.potential_path:
            potential = self.params.potential_path
        elif self.config.eon and self.config.eon.potential_path:
            potential = self.config.eon.potential_path

        if not potential:
            msg = "Potential path not found in EON config or scenario parameters."
            raise ValueError(msg)

        if not potential.exists():
            msg = f"Potential file not found: {potential}"
            raise ValueError(msg)

        return potential

    def _run_akmc(self, structure: Atoms) -> None:
        """Runs adaptive KMC using EON."""
        if not self.config.eon:
            return

        # Use injected wrapper or create new one via factory/config if missing
        if not self.eon_wrapper:
            self.eon_wrapper = EONWrapper(self.config.eon)

        work_dir = Path("eon_work")
        work_dir.mkdir(exist_ok=True)

        try:
            # EON requires pos.con. ASE format 'eon' writes .con files.
            write(work_dir / "pos.con", structure, format="eon")
        except Exception as e:
            msg = f"Failed to write EON structure: {e}"
            logger.warning(
                "Failed to write pos.con using 'eon' format: %s. Trying fallback if possible.", e
            )
            # Try plain extended xyz as fallback or just fail
            with contextlib.suppress(Exception):
                write(work_dir / "pos.con", structure, format="extxyz")  # Might not work for EON
            raise RuntimeError(msg) from e

        # Generate config
        self.eon_wrapper.generate_config(work_dir / "config.ini")

        # Run EON
        try:
            self.eon_wrapper.run(work_dir)
        except RuntimeError:
            # Already logged in wrapper
            return

        # Parse results
        results = self.eon_wrapper.parse_results(work_dir)
        logger.info("aKMC Results Summary: %s", results.keys())
