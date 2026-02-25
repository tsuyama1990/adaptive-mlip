import contextlib
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

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
    potential_path: Path | None = Field(None, description="Path to potential file (overrides EON config)")
    mgo_lattice_constant: float = Field(4.212, gt=0.0, description="Lattice constant of MgO")
    deposition_height: float = Field(2.5, gt=0.0, description="Height above surface for deposition (Angstroms)")
    write_intermediate_files: bool = Field(True, description="Whether to write intermediate XYZ files")


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

        # 2. Deposit Fe/Pt Atoms using MD Engine
        # We work on the existing structure object to minimize copies,
        # but deposit_atoms returns the final structure.
        deposited_structure = self._deposit_atoms(mgo_surface)
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
        return surface(mgo, (0, 0, 1), layers=4, vacuum=10.0)

    def _deposit_atoms(self, slab: Atoms) -> Atoms:
        """
        Deposits Fe and Pt atoms onto the surface.
        Uses MD Engine for relaxation after placement.
        """
        num_depositions = self.params.num_depositions
        ratio = self.params.fe_pt_ratio
        max_retries = 3
        dep_height = self.params.deposition_height

        # Work on a copy initially, then reuse
        structure = slab.copy()  # type: ignore[no-untyped-call]

        # Determine potential path
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

        for i in range(num_depositions):
            # Choose element
            element = "Fe" if random.random() < ratio else "Pt"  # noqa: S311

            # Random position above surface
            # Find max z of surface atoms
            if len(structure) > 0:
                # Use get_positions() to be safe across ASE versions
                positions = structure.get_positions()
                max_z = positions[:, 2].max()
            else:
                max_z = 0.0

            # Random x, y within cell
            lx = structure.cell[0, 0]
            ly = structure.cell[1, 1]
            x = random.uniform(0, lx)  # noqa: S311
            y = random.uniform(0, ly)  # noqa: S311
            z = max_z + dep_height

            # Use Atom object to ensure position is set correctly
            atom = Atom(element, position=(x, y, z))
            structure.append(atom)

            # Relax structure using MD Engine with retries
            relaxed_structure = None
            for attempt in range(max_retries):
                try:
                    # Relax returns a new Atoms object usually (depending on engine impl)
                    # Ideally engine.relax modifies in place or returns efficient copy.
                    # LammpsEngine.relax returns a result structure.
                    # We update 'structure' to point to this new object.
                    # This replaces the old structure reference, allowing GC to reclaim if refcount drops.
                    # This is acceptable for incremental growth unless N is huge.
                    relaxed_structure = self.engine.relax(structure, potential)
                    break # Success
                except Exception as e:
                    logger.warning("Deposition %d relaxation failed (attempt %d/%d): %s", i, attempt + 1, max_retries, e)

            if relaxed_structure is not None:
                structure = relaxed_structure
            else:
                logger.error("Failed to relax structure after deposition %d. Aborting deposition loop.", i)
                msg = f"Deposition failed at step {i} after {max_retries} attempts."
                raise RuntimeError(msg)

        return structure

    def _run_akmc(self, structure: Atoms) -> None:
        """Runs adaptive KMC using EON."""
        if not self.config.eon:
            return

        # Use injected wrapper or create new one
        wrapper = self.eon_wrapper or EONWrapper(self.config.eon)

        work_dir = Path("eon_work")
        work_dir.mkdir(exist_ok=True)

        try:
            # EON requires pos.con. ASE format 'eon' writes .con files.
            write(work_dir / "pos.con", structure, format="eon")
        except Exception as e:
            msg = f"Failed to write EON structure: {e}"
            logger.warning("Failed to write pos.con using 'eon' format: %s. Trying fallback if possible.", e)
            # Try plain extended xyz as fallback or just fail
            with contextlib.suppress(Exception):
                write(work_dir / "pos.con", structure, format="extxyz") # Might not work for EON
            raise RuntimeError(msg) from e

        # Generate config
        wrapper.generate_config(work_dir / "config.ini")

        # Run EON
        try:
            wrapper.run(work_dir)
        except RuntimeError:
            # Already logged in wrapper
            return

        # Parse results
        results = wrapper.parse_results(work_dir)
        logger.info("aKMC Results Summary: %s", results.keys())
