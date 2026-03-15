from pathlib import Path

content = Path("src/pyacemaker/utils/spatial.py").read_text()
if (
    "ACTION_PRIORITY" in content
    and "from pyacemaker.domain_models.defaults import ACTION_PRIORITY" not in content
):
    content = content.replace(
        "ACTION_PRIORITY: dict[SpatialAction, int] = {",
        "# ACTION_PRIORITY moved to defaults\nfrom pyacemaker.domain_models.defaults import ACTION_PRIORITY\n# ACTION_PRIORITY: dict[SpatialAction, int] = {",
    )
    content = content.replace(
        "    SpatialAction.ACTION_ACTIVE_LEARNING_ONLY: 1,\n    SpatialAction.ACTION_LANGEVIN_THERMOSTAT: 2,\n    SpatialAction.ACTION_FREEZE: 3,\n}",
        "# }",
    )
    Path("src/pyacemaker/utils/spatial.py").write_text(content)
