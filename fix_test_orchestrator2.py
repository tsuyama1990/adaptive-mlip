import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'config.distillation.step1_direct_sampling.target_points = 100',
    'config.distillation.step1_direct_sampling.target_points = 100\n    config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]'
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)
