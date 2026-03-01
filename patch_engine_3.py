with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

# Remove unused ignores and fix type
content = content.replace("forces = driver.get_forces()  # type: ignore[assignment]", "forces = driver.get_forces()")
content = content.replace("forces: list[list[float]] = driver.get_forces()  # type: ignore[assignment]", "forces = driver.get_forces()")
content = content.replace("stress: list[float] = list(stress_array)  # type: ignore[arg-type]", "stress = list(stress_array)")
content = content.replace("stress = list(stress_array) # type: ignore", "stress = list(stress_array)")
content = content.replace("return driver.get_atoms(elements) # type: ignore[no-any-return]", "return driver.get_atoms(elements)")

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
