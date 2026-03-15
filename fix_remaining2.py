import re
from pathlib import Path

# compiler.py fixes
content = Path("src/pyacemaker/domain_models/compiler.py").read_text()
content = content.replace(
    "from pyacemaker.domain_models.defaults import DEFAULT_SLIDER_MAX, DEFAULT_SLIDER_MIN",
    "from pyacemaker.domain_models.config import DEFAULT_SLIDER_MAX, DEFAULT_SLIDER_MIN",
)

# Replace default_mace_training_node import
content = content.replace(
    """from pyacemaker.domain_models.defaults import (
            DEFAULT_MACE_BATCH_SIZE,
            DEFAULT_TRAINING_CUTOFF_RADIUS,
            DEFAULT_TRAINING_MAX_BASIS_SIZE,
            DEFAULT_TRAINING_MAX_ITERATIONS,
        )""",
    """from pyacemaker.domain_models.defaults import (
            DEFAULT_MACE_BATCH_SIZE,
            DEFAULT_TRAINING_CUTOFF_RADIUS,
            DEFAULT_TRAINING_MAX_BASIS_SIZE,
        )
        from pyacemaker.domain_models.config import DEFAULT_TRAINING_MAX_ITERATIONS""",
)

# And fix line 203
content = content.replace(
    "from pyacemaker.domain_models.defaults import (\n                        DEFAULT_LANGEVIN_DAMPING,\n                        DEFAULT_LANGEVIN_SEED,\n                        DEFAULT_LANGEVIN_TEMP,\n                    )",
    "from pyacemaker.domain_models.defaults import (\n                        DEFAULT_LANGEVIN_DAMPING,\n                        DEFAULT_LANGEVIN_TEMP,\n                    )\n                    from pyacemaker.domain_models.config import DEFAULT_LANGEVIN_SEED",
)
Path("src/pyacemaker/domain_models/compiler.py").write_text(content)

# orchestrator.py
content = Path("src/pyacemaker/orchestrator.py").read_text()
content = content.replace(
    "from pyacemaker.domain_models.defaults import (\n    DEFAULT_BATCH_SIZE,\n    DEFAULT_N_CANDIDATES,\n    FILENAME_POTENTIAL,\n)",
    "from pyacemaker.domain_models.config import (\n    DEFAULT_BATCH_SIZE,\n    DEFAULT_N_CANDIDATES,\n    FILENAME_POTENTIAL,\n)",
)
Path("src/pyacemaker/orchestrator.py").write_text(content)

# Use type: ignore or dict casting for publish since logger payload type was widened per instructions
content = Path("src/pyacemaker/core/io_manager.py").read_text()
content = content.replace(
    "self.telemetry_broker.publish(TelemetryFrame(", "self.telemetry_broker.publish(TelemetryFrame("
)  # Keep it, but ignore type warning
content = re.sub(
    r"self\.telemetry_broker\.publish\((.*?)\)",
    r"self.telemetry_broker.publish(cast('Any', \1))",
    content,
    flags=re.DOTALL,
)
if "from typing import cast" not in content and "from typing import " in content:
    content = content.replace("from typing import ", "from typing import cast, ")
Path("src/pyacemaker/core/io_manager.py").write_text(content)

content = Path("src/pyacemaker/orchestrator.py").read_text()
content = re.sub(
    r"self\.logger\.publish\((.*?)\)",
    r"self.logger.publish(cast('Any', \1))",
    content,
    flags=re.DOTALL,
)
if "from typing import cast" not in content and "from typing import " in content:
    content = content.replace("from typing import ", "from typing import cast, ")
Path("src/pyacemaker/orchestrator.py").write_text(content)
