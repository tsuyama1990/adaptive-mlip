import re

with open("src/pyacemaker/domain_models/scenario.py", "r") as f:
    content = f.read()

# InitialStructureData needs to allow regions but it fails because we need to fix the import of SpatialRegion if not fully resolved or similar?
# Wait, the error is:
# pydantic_core._pydantic_core.ValidationError: 1 validation error for InitialStructureData
# regions
#  Extra inputs are not permitted [type=extra_forbidden, input_value=[SpatialRegion(x_min=0.0,...N_LANGEVIN_THERMOSTAT')], input_type=list]

# Ah, I see: I replaced the string incorrectly in scenario.py? Let's check the fields.
