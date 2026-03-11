import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Add src to path to allow importing pyacemaker
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.dft import DFTConfig
from pyacemaker.domain_models.logging import LoggingConfig
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.domain_models.validation import ValidationConfig
from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    CutoutConfig,
    DistillationConfig,
    LoopStrategyConfig,
    OTFConfig,
    WorkflowConfig,
)
from pyacemaker.orchestrator import Orchestrator


def fake_run_command(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
    if "pace_train" in cmd:
        # simulate pace_train output
        out_path = Path(cmd[1]).parent / "potential.yace"
        out_path.touch()
        return
    return

def main() -> int:
    # 1. Setup Environment
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()
    base_dir = Path(temp_dir.name).resolve()
    os.chdir(base_dir)

    potentials_dir = base_dir / "potentials"
    potentials_dir.mkdir(parents=True, exist_ok=True)
    mace_model_file = potentials_dir / "mace-mp-0-medium.model"
    mace_model_file.touch()

    for el in ["Mg", "O"]:
        (base_dir / f"{el}.UPF").touch()

    # 2. Configuration Definition
    config = PyAceConfig(
        project_name="UAT_01_Zero_Shot",
        structure=StructureConfig(
            elements=["Mg", "O"],
            supercell_size=[2, 2, 2],
        ),
        dft=DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=400,
            pseudopotentials={"Mg": "Mg.UPF", "O": "O.UPF"},
        ),
        training=TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=500,
        ),
        md=MDConfig(
            temperature=300.0,
            pressure=0.0,
            timestep=0.001,
            n_steps=1000,
        ),
        validation=ValidationConfig(),
        logging=LoggingConfig(level="INFO"),
        workflow=WorkflowConfig(
            max_iterations=1,
            batch_size=5,
            n_candidates=10,
            data_dir=str(base_dir / "data"),
            active_learning_dir=str(base_dir / "active_learning"),
            potentials_dir=str(base_dir / "potentials"),
            distillation=DistillationConfig(
                enable=True,
                mace_model_path=str(mace_model_file),
                uncertainty_threshold=0.5,
                sampling_structures_per_system=20,
            ),
            loop_strategy=LoopStrategyConfig(
                use_tiered_oracle=True,
                incremental_update=True,
                replay_buffer_size=500,
                baseline_potential_type="LJ",
                thresholds=ActiveLearningThresholds(
                    threshold_call_dft=0.05,
                    threshold_add_train=0.02,
                    smooth_steps=3,
                )
            ),
            cutout=CutoutConfig(
                core_radius=4.0,
                buffer_radius=3.0,
                enable_pre_relaxation=True,
                enable_passivation=True,
                passivation_element="H",
            ),
            otf=OTFConfig()
        )
    )

    # 3. Execution of Phase 1 (Zero-Shot Initialization)
    import io
    import logging
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    try:
        # Use patch to mock external commands instead of dropping bash files
        with patch("pyacemaker.core.trainer.shutil.which", return_value="/fake/path"),              patch("pyacemaker.core.trainer.run_command", side_effect=fake_run_command):

            orchestrator = Orchestrator(config)
            orchestrator.initialize_modules()
            orchestrator._check_initial_potential()

        logs = log_capture.getvalue()

        if "Total calls made to the DFTManager during this entire iteration: 0" not in logs:
            return 1

        if "Zero-Shot Distillation enabled. Generating 20 combinatorial structures" not in logs:
            return 1

        if not (base_dir / "active_learning" / "iter_000" / "training" / "potential.yace").exists():
            return 1

    except Exception:
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
