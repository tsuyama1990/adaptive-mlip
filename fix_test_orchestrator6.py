import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'config.distillation.step1_direct_sampling.descriptor.method = "SOAP"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5',
    'config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5'
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)

with open('src/pyacemaker/utils/descriptors.py', 'r') as f:
    content = f.read()

content = content.replace(
    'method = self.config.method.lower()',
    'method = getattr(self.config, "method", "soap")\n        if not isinstance(method, str):\n            method = "soap"\n        method = method.lower()'
)

with open('src/pyacemaker/utils/descriptors.py', 'w') as f:
    f.write(content)
