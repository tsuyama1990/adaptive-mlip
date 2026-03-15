from pathlib import Path

# Move import in defaults.py
content = Path("src/pyacemaker/domain_models/defaults.py").read_text()
if "from pyacemaker.domain_models.scenario import SpatialAction" in content and "from typing import Final" in content:
    content = content.replace("from pyacemaker.domain_models.scenario import SpatialAction", "")
    content = content.replace("from typing import Final", "from typing import Final\nfrom pyacemaker.domain_models.scenario import SpatialAction")
    Path("src/pyacemaker/domain_models/defaults.py").write_text(content)

# Move import in spatial.py
content = Path("src/pyacemaker/utils/spatial.py").read_text()
if "from pyacemaker.domain_models.defaults import ACTION_PRIORITY" in content:
    content = content.replace("from pyacemaker.domain_models.defaults import ACTION_PRIORITY", "")
    content = content.replace("from pyacemaker.domain_models.scenario import SpatialRegion", "from pyacemaker.domain_models.scenario import SpatialRegion\nfrom pyacemaker.domain_models.defaults import ACTION_PRIORITY")
    content = content.replace("# ACTION_PRIORITY: dict[SpatialAction, int] = {\n# }", "")
    Path("src/pyacemaker/utils/spatial.py").write_text(content)
