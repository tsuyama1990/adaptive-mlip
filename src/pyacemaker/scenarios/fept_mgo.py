import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.io import write
from ase.visualize.plot import plot_atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.interfaces.eon_driver import EONWrapper

from .base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class FePtMgoScenario(BaseScenario):
    """Scenario for Fe/Pt on MgO deposition and ordering."""

    def __init__(self, config: PyAceConfig) -> None:
        super().__init__(config)
        self.name = "fept_mgo"

        # Determine output directory
        out_dir = "fept_mgo_run"
        if self.config.scenario and self.config.scenario.output_dir:
            out_dir = self.config.scenario.output_dir
        self.work_dir = Path(out_dir)

        # Initialize LammpsEngine
        self.engine = LammpsEngine(config.md)

        # Initialize EONWrapper
        if config.eon:
            self.eon = EONWrapper(config.eon)
        else:
            self.eon = None

    def run(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting FePt/MgO Scenario in {self.work_dir}")

        # 1. Surface
        slab = self.generate_surface()

        # 2. Deposition
        deposited = self.deposit_atoms(slab)

        # 3. Ordering (kMC)
        ordered = deposited
        if self.eon and self.config.eon.enabled:
            ordered = self.run_akmc(deposited)
        else:
            logger.info("EON disabled or not configured. Skipping kMC.")

        # 4. Visualize
        self.visualize(ordered)

    def generate_surface(self) -> Atoms:
        logger.info("Generating MgO (001) Surface...")
        mgo = bulk("MgO", "rocksalt", a=4.21)

        size = (2, 2, 1)
        if self.config.scenario:
            size = self.config.scenario.slab_size

        slab = surface(mgo, (0, 0, 1), 2)
        slab = slab.repeat(size)
        slab.center(vacuum=10.0, axis=2)

        output_path = self.work_dir / "mgo_slab.xyz"
        write(output_path, slab)
        logger.info(f"Surface generated: {output_path}")
        return slab

    def deposit_atoms(self, slab: Atoms) -> Atoms:
        logger.info("Depositing Fe/Pt atoms...")

        count = 10
        if self.config.scenario:
            count = self.config.scenario.deposition_count

        deposited = slab.copy()  # type: ignore[no-untyped-call]
        rng = np.random.default_rng(12345)

        z_height = slab.positions[:, 2].max() + 3.0

        for i in range(count):
            species = "Fe" if i % 2 == 0 else "Pt"
            x = rng.uniform(0, slab.cell[0, 0])
            y = rng.uniform(0, slab.cell[1, 1])
            z = z_height + rng.uniform(0, 2.0)
            deposited.append(species)
            deposited.positions[-1] = [x, y, z]

        # Relax
        is_mock = self.config.scenario and self.config.scenario.mock

        if is_mock:
            logger.info("Mock mode: Skipping relaxation.")
        else:
            # We need a potential path
            pot_path = self.config.md.potential_path
            if pot_path:
                try:
                    logger.info(f"Relaxing structure using potential: {pot_path}")
                    deposited = self.engine.relax(deposited, pot_path)
                except Exception as e:
                    logger.warning(f"Relaxation failed: {e}. Proceeding with unrelaxed.")
            else:
                logger.warning("No potential path in config.md. Skipping relaxation.")

        output_path = self.work_dir / "deposited.xyz"
        write(output_path, deposited)
        logger.info(f"Deposited structure saved: {output_path}")
        return deposited

    def run_akmc(self, atoms: Atoms) -> Atoms:
        logger.info("Running aKMC with EON...")

        reactant_path = self.work_dir / "reactant.con"
        pos_path = self.work_dir / "pos.con"

        # Write EON format
        write(reactant_path, atoms, format="eon")
        write(pos_path, atoms, format="eon")

        is_mock = self.config.scenario and self.config.scenario.mock

        if is_mock:
            logger.info("Mock mode: EON run simulated.")
            (self.work_dir / "processtable.dat").write_text("0 0.5 1.0\n1 0.6 1.0")
        else:
            pot_path = self.config.eon.potential_path or self.config.md.potential_path
            if not pot_path:
                logger.error("No potential path found for EON.")
                msg = "Potential path required for EON."
                raise ValueError(msg)

            # Use self.eon which is verified to be not None in run() check
            if self.eon:
                elements = sorted({*atoms.get_chemical_symbols()})
                self.eon.run(pot_path, self.work_dir, elements=elements)

        return atoms

    def visualize(self, atoms: Atoms) -> None:
        try:
            fig, ax = plt.subplots()
            plot_atoms(atoms, ax, radii=0.5, rotation=("10x,10y,0z"))
            fig.savefig(self.work_dir / "final_structure.png")
            plt.close(fig)
            logger.info("Visualization saved.")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
