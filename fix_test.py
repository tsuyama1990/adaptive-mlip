import re

with open('tests/unit/test_domain_models_distillation.py', 'r') as f:
    content = f.read()

content = content.replace(
    'config = Step1DirectSamplingConfig()',
    'config = Step1DirectSamplingConfig(descriptor={"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1})'
)
content = content.replace(
    'Step1DirectSamplingConfig(target_points=-1)',
    'Step1DirectSamplingConfig(target_points=-1, descriptor={"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1})'
)
content = content.replace(
    'config = DistillationConfig()',
    'config = DistillationConfig(enable_mace_distillation=False)'
)
content = content.replace(
    'assert isinstance(config.step1_direct_sampling, Step1DirectSamplingConfig)',
    'assert config.step1_direct_sampling is None'
)
content = content.replace(
    'assert isinstance(config.step2_active_learning, Step2ActiveLearningConfig)',
    'assert config.step2_active_learning is None'
)

new_override = """def test_distillation_config_override() -> None:
    config = DistillationConfig(
        enable_mace_distillation=True,
        step1_direct_sampling={"target_points": 500, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
        step2_active_learning={"uncertainty_threshold": 0.5},
        step3_mace_finetune={"base_model": "test"},
    )
    assert config.enable_mace_distillation is True
    assert config.step1_direct_sampling.target_points == 500
    assert config.step2_active_learning.uncertainty_threshold == 0.5"""

content = re.sub(r'def test_distillation_config_override\(\) -> None:.*?(?=def test_distillation_config_logic_validation\(\) -> None:)', new_override + '\n\n\n', content, flags=re.DOTALL)


new_logic = """def test_distillation_config_logic_validation() -> None:
    \"\"\"Test custom logic validator.\"\"\"
    with pytest.raises(ValidationError, match="Step 1 target points must be at least 10"):
        DistillationConfig(
            enable_mace_distillation=True,
            step1_direct_sampling={"target_points": 5, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
            step2_active_learning={"uncertainty_threshold": 0.5},
            step3_mace_finetune={"base_model": "test"},
        )"""

content = re.sub(r'def test_distillation_config_logic_validation\(\) -> None:.*?(?=def test_distillation_config_logic_validation_disabled\(\) -> None:)', new_logic + '\n\n\n', content, flags=re.DOTALL)


new_logic_disabled = """def test_distillation_config_logic_validation_disabled() -> None:
    \"\"\"Validator should skip if disabled.\"\"\"
    with pytest.raises(ValidationError, match="Distillation step configs must be None when enable_mace_distillation is False"):
        DistillationConfig(
            enable_mace_distillation=False,
            step1_direct_sampling={"target_points": 5, "descriptor": {"method": "soap", "species": ["H"], "r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.1}},
        )"""
content = re.sub(r'def test_distillation_config_logic_validation_disabled\(\) -> None:.*', new_logic_disabled + '\n', content, flags=re.DOTALL)


with open('tests/unit/test_domain_models_distillation.py', 'w') as f:
    f.write(content)
