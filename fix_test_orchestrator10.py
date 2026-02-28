import re

with open('tests/unit/test_orchestrator_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    '# Add proper config mocking\n    config.distillation.step1_direct_sampling.descriptor.method = "soap"\n    config.distillation.step1_direct_sampling.descriptor.r_cut = 5.0\n    config.distillation.step1_direct_sampling.descriptor.n_max = 8\n    config.distillation.step1_direct_sampling.descriptor.l_max = 6\n    config.distillation.step1_direct_sampling.descriptor.sigma = 0.5\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]\n    config.distillation.step1_direct_sampling.descriptor.sparse = False\n    # Mock the generator and its iteration to prevent trying to generate\n    config.distillation.step1_direct_sampling.target_points = 1\n    config.distillation.step1_direct_sampling.descriptor.species = ["H"]',
    '# Add proper config mocking\n    from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig\n    from pyacemaker.domain_models.active_learning import DescriptorConfig\n    config.distillation.step1_direct_sampling = Step1DirectSamplingConfig(\n        target_points=1,\n        descriptor=DescriptorConfig(method="soap", species=["H"], r_cut=5.0, n_max=2, l_max=2, sigma=0.1)\n    )'
)


with open('tests/unit/test_orchestrator_distillation.py', 'w') as f:
    f.write(content)
