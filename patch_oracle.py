import re

with open("src/pyacemaker/core/oracle.py", "r") as f:
    content = f.read()

# Add Protocol
protocol = """
from typing import Protocol, Any, Dict

class CalculatorProtocol(Protocol):
    results: Dict[str, Any]
    def calculate(self, atoms: Atoms, properties: list[str], system_changes: list[str]) -> None: ...
    def get_property(self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True) -> Any: ...

"""

content = content.replace("from typing import Any", protocol)

mace_import = """
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    mace_mp = None
"""

content = content.replace("from pyacemaker.core.exceptions import OracleError", "from pyacemaker.core.exceptions import OracleError\n" + mace_import)

# Update MACEManager
manager = """
class MACEManager(BaseOracle):
    \"\"\"
    A wrapper around the MACE Python package for fast structure evaluation and uncertainty estimation.
    \"\"\"

    def __init__(self, model_path: str = "mace-mp-0-medium") -> None:
        self.model_path = model_path
        self._calculator: CalculatorProtocol | None = None

    @property
    def calculator(self) -> CalculatorProtocol:
        if self._calculator is None:
            if not MACE_AVAILABLE:
                msg = "The 'mace' package is required for MACEManager. Please install it with 'pip install mace-torch'"
                raise RuntimeError(msg)

            # Use mace_mp(model=self.model_path) if available, else a dummy or standard init.
            self._calculator = mace_mp(model=self.model_path)  # type: ignore[misc]
        return self._calculator
"""

content = re.sub(r'class MACEManager\(BaseOracle\):.*?return self\._calculator', manager, content, flags=re.DOTALL)

with open("src/pyacemaker/core/oracle.py", "w") as f:
    f.write(content)
