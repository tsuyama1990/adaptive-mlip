import re

with open("src/pyacemaker/domain_models/compiler.py", "r") as f:
    content = f.read()

# Let's inspect `_apply_spatial_logic`. Did it assign it to md_config?
# Wait! In `_apply_spatial_logic`, it does `md_config.custom_initialization_commands = cmds`. But `MDConfig` is returned, then PyAceConfig is built.
# Let's look at `compiler.py` `_apply_spatial_logic`.
