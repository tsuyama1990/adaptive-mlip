import re

with open("src/pyacemaker/core/oracle.py", "r") as f:
    content = f.read()

# Fix module level imports
# The issue is that we injected `from typing import Protocol...` at the top of the file before `from ase import Atoms`.
# Let's cleanly reconstruct the top of oracle.py
top_imports = """import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Dict, Protocol

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from pyacemaker.core.base import BaseOracle
from pyacemaker.core.exceptions import OracleError
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.constants import ERR_ORACLE_FAILED, ERR_ORACLE_ITERATOR
from pyacemaker.interfaces.qe_driver import QEDriver
from pyacemaker.utils.embedding import embed_cluster

try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    mace_mp = None

class CalculatorProtocol(Protocol):
    results: Dict[str, Any]
    def calculate(self, atoms: Atoms, properties: list[str], system_changes: list[str]) -> None: ...
    def get_property(self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True) -> Any: ...

logger = logging.getLogger(__name__)
"""

# replace everything before `class SelfHealingManager`
content = re.sub(r'^.*?class SelfHealingManager:', top_imports + '\nclass SelfHealingManager:', content, flags=re.DOTALL)

# Fix Unused "type: ignore" comment in MACEManager
content = content.replace("self._calculator = mace_mp(model=self.model_path)  # type: ignore[misc]", "self._calculator = mace_mp(model=self.model_path)")

with open("src/pyacemaker/core/oracle.py", "w") as f:
    f.write(content)
