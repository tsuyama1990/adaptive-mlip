import re

with open("tests/conftest.py", "r") as f:
    content = f.read()

content = content.replace(
    "def create_test_config_dict(**overrides: Any) -> dict[str, Any]:",
    """
class StructureDict(TypedDict, total=False):
    elements: list[str]
    supercell_size: list[int]
    policy_name: str

class DFTDict(TypedDict, total=False):
    code: str
    functional: str
    kpoints_density: float
    encut: float
    pseudopotentials: dict[str, str]
    mixing_beta: float
    smearing_type: str
    smearing_width: float
    diagonalization: str

class TrainingDict(TypedDict, total=False):
    potential_type: str
    cutoff_radius: float
    max_basis_size: int
    delta_learning: bool
    active_set_optimization: bool

class MDDict(TypedDict, total=False):
    temperature: float
    pressure: float
    timestep: float
    n_steps: int
    uncertainty_threshold: float
    check_interval: int

class OTFDict(TypedDict, total=False):
    uncertainty_threshold: float
    local_n_candidates: int
    local_n_select: int
    max_retries: int

class WorkflowDict(TypedDict, total=False):
    max_iterations: int
    state_file_path: str
    active_learning_dir: str
    potentials_dir: str
    n_candidates: int
    batch_size: int
    otf: OTFDict

class ConfigDictType(TypedDict, total=False):
    project_name: str
    structure: StructureDict
    dft: DFTDict
    training: TrainingDict
    md: MDDict
    validation: dict[str, Any]
    workflow: WorkflowDict
    logging: dict[str, Any]

def create_test_config_dict(**overrides: Any) -> ConfigDictType:
""",
)

with open("tests/conftest.py", "w") as f:
    f.write(content)
