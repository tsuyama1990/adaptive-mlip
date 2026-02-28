import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]',
    'config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]\n    config.distillation.step1_direct_sampling.descriptor.sparse = False'
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)
