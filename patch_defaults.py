from pathlib import Path

content = Path("src/pyacemaker/domain_models/defaults.py").read_text()

additions = """
# Heuristic Fallback Constants
DEFAULT_SMEARING_TYPE = "gaussian"
DEFAULT_SMEARING_WIDTH = 0.1
ELEMENT_SMEARING_FALLBACKS = {
    "Pt": {"smearing_type": "mv", "smearing_width": 0.02}
}

# Active Learning Heuristic Configs
DEFAULT_HEURISTIC_DFT_THRESHOLD = 0.05
DEFAULT_HEURISTIC_MD_TIMESTEP = 0.001
DEFAULT_HEURISTIC_CHECK_INTERVAL = 10
DEFAULT_HEURISTIC_ENCUT = 40.0
DEFAULT_HEURISTIC_LEARNING_RATE = 0.01
"""

if "ELEMENT_SMEARING_FALLBACKS" not in content:
    content += "\n" + additions
    Path("src/pyacemaker/domain_models/defaults.py").write_text(content)
