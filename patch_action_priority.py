from pathlib import Path

content = Path("src/pyacemaker/domain_models/defaults.py").read_text()
if "ACTION_PRIORITY" not in content:
    additions = """
# Action Priority
from pyacemaker.domain_models.scenario import SpatialAction
ACTION_PRIORITY: dict[SpatialAction, int] = {
    SpatialAction.ACTION_ACTIVE_LEARNING_ONLY: 1,
    SpatialAction.ACTION_LANGEVIN_THERMOSTAT: 2,
    SpatialAction.ACTION_FREEZE: 3,
}
"""
    content += "\n" + additions
    Path("src/pyacemaker/domain_models/defaults.py").write_text(content)
