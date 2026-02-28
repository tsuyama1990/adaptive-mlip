import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]\n    config.distillation.step1_direct_sampling.descriptor.sparse = False',
    '# Add proper config mocking\n    config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]\n    config.distillation.step1_direct_sampling.descriptor.sparse = False\n    # Mock the generator and its iteration to prevent trying to generate\n    config.distillation.step1_direct_sampling.target_points = 1'
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)

with open('src/pyacemaker/orchestrator.py', 'r') as f:
    content = f.read()

content = content.replace(
    'sampler = DirectSampler(config, self.generator)',
    'sampler = DirectSampler(config, self.generator)  # type: ignore[arg-type]'
)

with open('src/pyacemaker/orchestrator.py', 'w') as f:
    f.write(content)
