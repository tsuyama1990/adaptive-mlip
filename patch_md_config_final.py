with open("src/pyacemaker/domain_models/md.py", "r") as f:
    content = f.read()

# I apparently didn't actually add it or it got lost. Let's add it properly.
replacement = """    soft_start_langevin_damp: float = Field(
        0.1, gt=0.0, description="Damping parameter (ps) for soft start Langevin thermostat"
    )

    custom_initialization_commands: list[str] = Field(
        default_factory=list, description="Custom LAMMPS initialization commands"
    )"""

content = content.replace("""    soft_start_langevin_damp: float = Field(
        0.1, gt=0.0, description="Damping parameter (ps) for soft start Langevin thermostat"
    )""", replacement)

with open("src/pyacemaker/domain_models/md.py", "w") as f:
    f.write(content)
