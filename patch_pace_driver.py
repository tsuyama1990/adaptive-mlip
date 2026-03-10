with open("src/pyacemaker/domain_models/defaults.py", "r") as f:
    content = f.read()

import re
content = re.sub(r'PACE_DRIVER_TEMPLATE = """.*?"""', '', content, flags=re.DOTALL)

with open("src/pyacemaker/domain_models/defaults.py", "w") as f:
    f.write(content)

with open("src/pyacemaker/domain_models/constants.py", "r") as f:
    content = f.read()
content = content.replace("    PACE_DRIVER_TEMPLATE,\n", "")
content = content.replace('    "PACE_DRIVER_TEMPLATE",\n', "")
with open("src/pyacemaker/domain_models/constants.py", "w") as f:
    f.write(content)

with open("src/pyacemaker/interfaces/eon_driver.py", "r") as f:
    content = f.read()

content = content.replace("from pyacemaker.domain_models.constants import PACE_DRIVER_TEMPLATE", "from pyacemaker.interfaces.pace_driver_template import PACE_DRIVER_TEMPLATE")

with open("src/pyacemaker/interfaces/eon_driver.py", "w") as f:
    f.write(content)

with open("src/pyacemaker/interfaces/pace_driver_template.py", "r") as f:
    content = f.read()

# We need to expose it as a string for eon_driver to write.
wrapper = 'PACE_DRIVER_TEMPLATE = """\n' + content + '"""\n'
with open("src/pyacemaker/interfaces/pace_driver_template.py", "w") as f:
    f.write(wrapper)
