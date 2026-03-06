import re
from pathlib import Path

# Provide defaults for supported_formats in Pydantic schema using proper syntax so it passes all unit tests
p_train_config = Path("src/pyacemaker/domain_models/training.py")
config_content = p_train_config.read_text()
# Ensure we define the default via default_factory
if "supported_formats: list[str] = Field(default_factory=list)" not in config_content:
    config_content = re.sub(
        r'supported_formats: list\[str\] = Field\(default_factory=lambda: \[\"\.pckl\", \"\.xyz\", \"\.extxyz\", \"\.gzip\"\], description=\"Allowed formats\"\)',
        'supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed formats")',
        config_content
    )
p_train_config.write_text(config_content)

# Instead of altering tests one by one, the easiest way to fix Pydantic tests failing on an unexpected attribute is to set `extra="ignore"`
# temporarily or just fix the initialization in tests. But since Pydantic schema is correct, let's fix the schema properly.
config_content = config_content.replace(
    'supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed formats")',
    'supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")'
)
p_train_config.write_text(config_content)
