from pathlib import Path

content = Path("tests/uat/test_cycle05_uat.py").read_text()

# Fix MDConfig strict kwargs
content = content.replace(
    "MDConfig(\n        n_steps=2000,\n        fix_halt=False,\n        temperature=300.0,\n        pressure=1.0,\n        timestep=0.001,\n        soft_start_steps=150,\n        soft_start_langevin_damp=0.2,\n    )",
    "MDConfig.model_validate({\n        'n_steps': 2000,\n        'fix_halt': False,\n        'temperature': 300.0,\n        'pressure': 1.0,\n        'timestep': 0.001,\n        'soft_start_steps': 150,\n        'soft_start_langevin_damp': 0.2,\n    })",
)

# Fix IntentRequest kwargs
content = content.replace("IntentRequest(**payload)", "IntentRequest.model_validate(payload)")

Path("tests/uat/test_cycle05_uat.py").write_text(content)
