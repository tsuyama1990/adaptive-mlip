import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    "config.distillation.step1_direct_sampling.descriptor.method = \"soap\"",
    "config.distillation.step1_direct_sampling.descriptor.method = \"SOAP\""
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)
