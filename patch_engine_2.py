import re

with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

# Fix the LammpsDriver __enter__ issue. The original code used try...finally. Let's revert the `with LammpsDriver` back to try...finally in run() and relax() but fix the bug the auditor noted.
# "LammpsEngine creates new LammpsDriver instance without proper resource cleanup in case of exceptions... Location: src/pyacemaker/core/engine.py (Line 65-66)"
# Let's check lines around 65 in engine.py:

# It seems `forces = driver.get_forces()` might have lacked types, which I fixed.

# Let's revert `run` and `relax` to use try/finally correctly.
run_method = """
    def run(
        self, structure: Atoms | None, potential: Any, restart_file: Path | None = None
    ) -> MDSimulationResult:
        \"\"\"
        Runs the MD simulation. If restart_file is provided, resumes exactly from that state.
        \"\"\"
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            # Generate input script to file
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            input_script_path = temp_dir / "input.lmp"

            with input_script_path.open("w") as f:
                self.generator.write_script(
                    f, potential_path, data_file, dump_file, elements, restart_file
                )

            # Initialize Driver with unique log file and use try/finally for cleanup
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self.executor.execute_simulation(driver, input_script_path)
                return self.parser.parse_md_result(driver, dump_file, log_file)
            finally:
                if hasattr(driver, "close"):
                    driver.close()
"""
content = re.sub(r'    def run\([\s\S]*?return self.parser.parse_md_result\(driver, dump_file, log_file\)', run_method, content)

relax_method = """
    def relax(self, structure: Atoms, potential: Any) -> Atoms:
        \"\"\"
        Relaxes the structure to a local minimum using LAMMPS minimize.
        \"\"\"
        ctx, data_file, dump_file, log_file, elements, potential_path = (
            self._prepare_simulation_env(structure, potential)
        )

        with ctx:
            # Generate minimization script
            temp_dir = Path(ctx.name) if hasattr(ctx, "name") else data_file.parent
            script_path = temp_dir / "relax.lmp"

            with script_path.open("w") as f:
                self.generator.write_minimization_script(f, potential_path, data_file, elements)

            # Execute with try/finally
            driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
            try:
                self.executor.execute_simulation(driver, script_path)
                return driver.get_atoms(elements) # type: ignore[no-any-return]
            finally:
                if hasattr(driver, "close"):
                    driver.close()
"""
content = re.sub(r'    def relax\([\s\S]*?return driver.get_atoms\(elements\)', relax_method, content)

# Fix the list comprehension
content = content.replace("list(int(idx) for idx in epicenter_indices)", "[int(idx) for idx in epicenter_indices]")
# Remove unused type: ignore at line 66 or so. Wait, the auditor said line 65-66. Let's see what line 66 actually is.
# In `parse_md_result`, I added: forces: list[list[float]] = driver.get_forces() # type: ignore[assignment]
# Maybe that was line 66? Let's fix line 66 manually.
content = content.replace("forces: list[list[float]] = driver.get_forces()  # type: ignore[assignment]", "forces = driver.get_forces()  # type: ignore[assignment]")

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
