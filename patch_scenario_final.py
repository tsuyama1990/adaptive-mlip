with open("src/pyacemaker/domain_models/scenario.py", "r") as f:
    content = f.read()

import re
content = re.sub(
    r"class InitialStructureData\(BaseModel\):\n    model_config = ConfigDict\(extra=\"forbid\", strict=True\)\n    type: Literal\[NodeType\.INITIAL_STRUCTURE\] = Field\(NodeType\.INITIAL_STRUCTURE\)\n    chemical_symbol: str = Field\(\.\.\., description=\"The chemical symbol\"\)\n    lattice_constant: float = Field\(\.\.\., description=\"The lattice constant\"\)",
    """from pyacemaker.domain_models.gui_schema import SpatialRegion

class InitialStructureData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal[NodeType.INITIAL_STRUCTURE] = Field(NodeType.INITIAL_STRUCTURE)
    chemical_symbol: str = Field(..., description="The chemical symbol")
    lattice_constant: float = Field(..., description="The lattice constant")
    regions: list[SpatialRegion] | None = Field(default=None, description="Spatial regions mapped to this structure")""",
    content
)

with open("src/pyacemaker/domain_models/scenario.py", "w") as f:
    f.write(content)
