from pathlib import Path

# Fix constants.py imports
content = Path("src/pyacemaker/domain_models/constants.py").read_text()
content = content.replace(
    "DANGEROUS_PATH_CHARS,\n    DEFAULT_EON_EXECUTABLE,\n    DEFAULT_EON_SEED,\n    DEFAULT_LAMMPS_MINIMIZE_MAX_ITER,\n    DEFAULT_LAMMPS_MINIMIZE_STEPS,\n    DEFAULT_LAMMPS_VELOCITY_SEED,\n    DEFAULT_LJ_PARAMS,\n    DEFAULT_MC_SEED,\n    DEFAULT_MD_MINIMIZE_FTOL,\n    DEFAULT_MD_MINIMIZE_TOL,",
    "DEFAULT_EON_EXECUTABLE,\n    DEFAULT_LAMMPS_MINIMIZE_MAX_ITER,\n    DEFAULT_LAMMPS_MINIMIZE_STEPS,\n    DEFAULT_LAMMPS_VELOCITY_SEED,\n    DEFAULT_LJ_PARAMS,\n    DEFAULT_MD_MINIMIZE_FTOL,\n    DEFAULT_MD_MINIMIZE_TOL,",
)
content = (
    "from pyacemaker.domain_models.config import DANGEROUS_PATH_CHARS, DEFAULT_EON_SEED, DEFAULT_MC_SEED\n"
    + content
)
Path("src/pyacemaker/domain_models/constants.py").write_text(content)

# Fix workflow.py imports
content = Path("src/pyacemaker/domain_models/workflow.py").read_text()
content = content.replace("DEFAULT_BATCH_SIZE,\n", "")
content = content.replace("DEFAULT_N_CANDIDATES,\n", "")
content = (
    "from pyacemaker.domain_models.config import DEFAULT_BATCH_SIZE, DEFAULT_N_CANDIDATES\n"
    + content
)
Path("src/pyacemaker/domain_models/workflow.py").write_text(content)

# Fix md.py
content = Path("src/pyacemaker/domain_models/md.py").read_text()
content = content.replace("DEFAULT_MD_BASE_ENERGY,\n", "")
content = "from pyacemaker.domain_models.config import DEFAULT_MD_BASE_ENERGY\n" + content
Path("src/pyacemaker/domain_models/md.py").write_text(content)

# Fix heuristics.py
content = Path("src/pyacemaker/domain_models/heuristics.py").read_text()
content = content.replace(
    'smearing_type = ELEMENT_SMEARING_FALLBACKS[el]["smearing_type"]  # type: ignore[assignment]',
    'smearing_type = ELEMENT_SMEARING_FALLBACKS[el]["smearing_type"]',
)
content = content.replace(
    'smearing_width = ELEMENT_SMEARING_FALLBACKS[el]["smearing_width"]  # type: ignore[assignment]',
    'smearing_width = ELEMENT_SMEARING_FALLBACKS[el]["smearing_width"]',
)
Path("src/pyacemaker/domain_models/heuristics.py").write_text(content)

# Fix compiler.py
content = Path("src/pyacemaker/domain_models/compiler.py").read_text()
content = content.replace(
    "from pyacemaker.domain_models.defaults import DEFAULT_LANGEVIN_DAMPING, DEFAULT_LANGEVIN_SEED, DEFAULT_LANGEVIN_TEMP",
    "from pyacemaker.domain_models.defaults import DEFAULT_LANGEVIN_DAMPING, DEFAULT_LANGEVIN_TEMP\n                    from pyacemaker.domain_models.config import DEFAULT_LANGEVIN_SEED",
)

content = content.replace(
    "from pyacemaker.domain_models.defaults import (\n            DEFAULT_MACE_BATCH_SIZE,\n            DEFAULT_TRAINING_CUTOFF_RADIUS,\n            DEFAULT_TRAINING_MAX_BASIS_SIZE,\n            DEFAULT_TRAINING_MAX_ITERATIONS,\n        )",
    "from pyacemaker.domain_models.defaults import (\n            DEFAULT_MACE_BATCH_SIZE,\n            DEFAULT_TRAINING_CUTOFF_RADIUS,\n            DEFAULT_TRAINING_MAX_BASIS_SIZE,\n        )\n        from pyacemaker.domain_models.config import DEFAULT_TRAINING_MAX_ITERATIONS",
)

content = content.replace(
    """from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_CODE,
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_FUNCTIONAL,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
            DEFAULT_DFT_SMEARING_TYPE,
            DEFAULT_DFT_SMEARING_WIDTH,
            DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            DEFAULT_ENCUT_BASE,
            DEFAULT_ENCUT_FACTOR,
            DEFAULT_KPOINTS_DENSITY_BASE,
            DEFAULT_KPOINTS_DENSITY_FACTOR,
            DEFAULT_PSEUDOPOTENTIAL_MAPPING,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
        )""",
    """from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
        )
        from pyacemaker.domain_models.config import (
            DEFAULT_DFT_CODE,
            DEFAULT_DFT_FUNCTIONAL,
            DEFAULT_SMEARING_TYPE as DEFAULT_DFT_SMEARING_TYPE,
            DEFAULT_SMEARING_WIDTH as DEFAULT_DFT_SMEARING_WIDTH,
            DEFAULT_SMEARING_WIDTH as DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            DEFAULT_ENCUT_BASE,
            DEFAULT_ENCUT_FACTOR,
            DEFAULT_KPOINTS_DENSITY_BASE,
            DEFAULT_KPOINTS_DENSITY_FACTOR,
            DEFAULT_PSEUDOPOTENTIAL_MAPPING,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
        )""",
)

content = content.replace(
    "from pyacemaker.domain_models.defaults import DEFAULT_SLIDER_MAX, DEFAULT_SLIDER_MIN",
    "from pyacemaker.domain_models.config import DEFAULT_SLIDER_MAX, DEFAULT_SLIDER_MIN",
)
content = content.replace(
    """        from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
        )""",
    """        from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
            DEFAULT_DFT_SMEARING_WIDTH_FACTOR
        )""",
)
content = content.replace("DEFAULT_SMEARING_WIDTH as DEFAULT_DFT_SMEARING_WIDTH_FACTOR,", "")
content = content.replace(
    "from pyacemaker.domain_models.defaults import DEFAULT_MAX_ITERATIONS",
    "from pyacemaker.domain_models.config import DEFAULT_MAX_ITERATIONS",
)
Path("src/pyacemaker/domain_models/compiler.py").write_text(content)

# Undo the logger decoupling since it breaks model serialization (they wanted it decoupled, but it needs to just use standard BaseModel dumps if we keep generic type. Actually, the Auditor feedback said "use generic message types instead of specific telemetry models"). If we use Any, we should let the caller serialize, or call `.model_dump_json()` inside the try block if it hasattr it.
content = Path("src/pyacemaker/logger.py").read_text()
content = content.replace(
    "serialized_payload = payload.model_dump_json()",
    "serialized_payload = payload.model_dump_json() if hasattr(payload, 'model_dump_json') else json.dumps(payload)",
)
content = "import json\n" + content
Path("src/pyacemaker/logger.py").write_text(content)

# Revert Orchestrator decoupling active_set base class because it wasn't defined.
content = Path("src/pyacemaker/orchestrator.py").read_text()
content = content.replace(
    "from pyacemaker.core.base import (\n    BaseActiveSetSelector,",
    "from pyacemaker.core.base import (",
)
content = content.replace(
    "self.active_set_selector: BaseActiveSetSelector | None = None",
    "self.active_set_selector: Any = None",
)
content = content.replace(
    "BaseActiveSetSelector(config=self.config.training)",
    "ActiveSetSelector(config=self.config.training)",
)
content = content.replace(
    "from pyacemaker.domain_models.defaults import FILENAME_POTENTIAL",
    "from pyacemaker.domain_models.config import FILENAME_POTENTIAL",
)
if "from pyacemaker.core.active_set import ActiveSetSelector" not in content:
    content = content.replace(
        "from pyacemaker.core.base import",
        "from pyacemaker.core.active_set import ActiveSetSelector\nfrom pyacemaker.core.base import",
    )
Path("src/pyacemaker/orchestrator.py").write_text(content)
