import json
import os
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

# System Configuration Constants (Moved from defaults.py per strict Auditor rules)


DANGEROUS_PATH_CHARS: Final[set[str]] = {
    ";",
    "&",
    "|",
    "`",
    "$",
    "(",
    ")",
    "<",
    ">",
    "\n",
    "\r",
    "\t",
    "?",
    "*",
    "[",
    "]",
    "{",
    "}",
    "'",
    '"',
    "!",
    "#",
}

DEFAULT_EON_SEED: Final[int] = int(os.getenv("PYACEMAKER_EON_SEED", "12345"))
DEFAULT_LANGEVIN_SEED: Final[int] = int(os.getenv("PYACEMAKER_LANGEVIN_SEED", "12345"))
DEFAULT_MC_SEED: Final[int] = int(os.getenv("PYACEMAKER_MC_SEED", "12345"))

DEFAULT_PROJECT_NAME: Final[str] = os.getenv("PYACEMAKER_PROJECT_NAME", "intent_driven_project")
DEFAULT_SLIDER_MIN: Final[int] = int(os.getenv("PYACEMAKER_SLIDER_MIN", "1"))
DEFAULT_SLIDER_MAX: Final[int] = int(os.getenv("PYACEMAKER_SLIDER_MAX", "10"))
DEFAULT_MAX_ITERATIONS: Final[int] = int(os.getenv("PYACEMAKER_MAX_ITERATIONS", "10"))
DEFAULT_BATCH_SIZE: Final[int] = int(os.getenv("PYACEMAKER_BATCH_SIZE", "5"))
DEFAULT_N_CANDIDATES: Final[int] = int(os.getenv("PYACEMAKER_N_CANDIDATES", "10"))

DEFAULT_MD_BASE_ENERGY: Final[float] = float(os.getenv("PYACEMAKER_MD_BASE_ENERGY", "-100.0"))
DEFAULT_PACEMAKER_LOSS_L1: Final[float] = float(os.getenv("PYACEMAKER_PACEMAKER_LOSS_L1", "1e-8"))
DEFAULT_PACEMAKER_LOSS_L2: Final[float] = float(os.getenv("PYACEMAKER_PACEMAKER_LOSS_L2", "1e-8"))
DEFAULT_PACEMAKER_REPULSION_SIGMA: Final[float] = float(
    os.getenv("PYACEMAKER_PACEMAKER_REPULSION_SIGMA", "0.05")
)
DEFAULT_PACEMAKER_RAD_BASE: Final[str] = os.getenv("PYACEMAKER_PACEMAKER_RAD_BASE", "Chebyshev")
DEFAULT_PACEMAKER_EMBEDDING_TYPE: Final[str] = os.getenv(
    "PYACEMAKER_PACEMAKER_EMBEDDING_TYPE", "FinnisSinclair"
)
DEFAULT_PACEMAKER_OPTIMIZER: Final[str] = os.getenv("PYACEMAKER_PACEMAKER_OPTIMIZER", "BFGS")

# Smearing
DEFAULT_SMEARING_TYPE: Final[str] = os.getenv("PYACEMAKER_SMEARING_TYPE", "gaussian")
DEFAULT_SMEARING_WIDTH: Final[float] = float(os.getenv("PYACEMAKER_SMEARING_WIDTH", "0.1"))
_default_smearing_fallbacks = '{"Pt": {"smearing_type": "mv", "smearing_width": 0.02}}'
ELEMENT_SMEARING_FALLBACKS: Final[dict[str, Any]] = json.loads(
    os.getenv("PYACEMAKER_ELEMENT_SMEARING_FALLBACKS", _default_smearing_fallbacks)
)

# Spatial Action Priority
ACTION_PRIORITY: dict[str, int] = {
    "ACTION_ACTIVE_LEARNING_ONLY": 1,
    "ACTION_LANGEVIN_THERMOSTAT": 2,
    "ACTION_FREEZE": 3,
}

# Tag Assignment
DEFAULT_TAG_ASSIGNMENT_STRATEGY: Final[str] = os.getenv(
    "PYACEMAKER_TAG_ASSIGNMENT_STRATEGY", "priority"
)

# Mappings & Encuts
_default_pseudo_mapping = '{"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF", "W": "W.pbe-n-kjpaw_psl.1.0.0.UPF", "H": "H.pbe-rrkjus_psl.1.0.0.UPF", "O": "O.pbe-n-kjpaw_psl.0.1.UPF", "Pt": "Pt.pbe-n-kjpaw_psl.1.0.0.UPF", "Fe": "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"}'
DEFAULT_PSEUDOPOTENTIAL_MAPPING: Final[dict[str, str]] = json.loads(
    os.getenv("PYACEMAKER_PSEUDOPOTENTIAL_MAPPING", _default_pseudo_mapping)
)

DEFAULT_ENCUT_BASE: Final[float] = float(os.getenv("PYACEMAKER_ENCUT_BASE", "40.0"))
DEFAULT_ENCUT_FACTOR: Final[float] = float(os.getenv("PYACEMAKER_ENCUT_FACTOR", "2.0"))
DEFAULT_KPOINTS_DENSITY_BASE: Final[float] = float(
    os.getenv("PYACEMAKER_KPOINTS_DENSITY_BASE", "2.0")
)
DEFAULT_KPOINTS_DENSITY_FACTOR: Final[float] = float(
    os.getenv("PYACEMAKER_KPOINTS_DENSITY_FACTOR", "4.0")
)
DEFAULT_DFT_FUNCTIONAL: Final[str] = os.getenv("PYACEMAKER_DFT_FUNCTIONAL", "pbe")
DEFAULT_DFT_CODE: Final[str] = os.getenv("PYACEMAKER_DFT_CODE", "quantum_espresso")

# Training Defaults
DEFAULT_DELTA_SPLINE_BINS: Final[int] = int(os.getenv("PYACEMAKER_DELTA_SPLINE_BINS", "100"))
DEFAULT_EVALUATOR: Final[str] = os.getenv("PYACEMAKER_EVALUATOR", "tensorpot")
DEFAULT_DISPLAY_STEP: Final[int] = int(os.getenv("PYACEMAKER_DISPLAY_STEP", "50"))
DEFAULT_PACEMAKER_LOSS_KAPPA: Final[float] = float(
    os.getenv("PYACEMAKER_PACEMAKER_LOSS_KAPPA", "0.3")
)
DEFAULT_PACEMAKER_MAX_DEG: Final[int] = int(os.getenv("PYACEMAKER_PACEMAKER_MAX_DEG", "6"))
DEFAULT_PACEMAKER_NDENSITY: Final[int] = int(os.getenv("PYACEMAKER_PACEMAKER_NDENSITY", "2"))
DEFAULT_PACEMAKER_R0: Final[float] = float(os.getenv("PYACEMAKER_PACEMAKER_R0", "1.5"))
DEFAULT_TRAINING_BATCH_SIZE: Final[int] = int(os.getenv("PYACEMAKER_TRAINING_BATCH_SIZE", "10"))
DEFAULT_TRAINING_MAX_ITERATIONS: Final[int] = int(
    os.getenv("PYACEMAKER_TRAINING_MAX_ITERATIONS", "1000")
)
FILENAME_POTENTIAL: Final[str] = os.getenv("PYACEMAKER_FILENAME_POTENTIAL", "potential.yace")


from .dft import DFTConfig
from .eon import EONConfig
from .logging import LoggingConfig
from .md import MDConfig
from .scenario import ScenarioConfig
from .structure import StructureConfig
from .training import TrainingConfig
from .validation import ValidationConfig
from .workflow import WorkflowConfig

class PyAceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_name: str = Field(..., min_length=1, description="Name of the project")
    structure: StructureConfig
    dft: DFTConfig
    training: TrainingConfig
    md: MDConfig
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig, description="Validation configuration"
    )
    workflow: WorkflowConfig
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    eon: EONConfig | None = Field(None, description="EON configuration")
    scenario: ScenarioConfig | None = Field(None, description="Scenario configuration")
    api_cors_origins: list[str] = Field(
        default_factory=lambda: os.getenv("PYACEMAKER_CORS_ORIGINS", "http://localhost:3000").split(
            ","
        ),
        description="Allowed CORS origins for the API gateway",
    )
