import shlex
from pathlib import Path
from typing import TextIO

from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import LAMMPS_MIN_STYLE_CG
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.utils.path import validate_path_safe


class LammpsScriptGenerator:
    """
    Generates LAMMPS input scripts based on MDConfig.
    Follows Single Responsibility Principle by isolating script generation logic.
    Supports writing directly to a file-like object to handle large scripts efficiently.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config
        # Use lru_cache for methods instead of manual dict
        self._atomic_numbers_cache: dict[str, int] = {}

    def _get_atomic_number(self, symbol: str) -> int:
        """Cached atomic number lookup."""
        if symbol not in self._atomic_numbers_cache:
            self._atomic_numbers_cache[symbol] = atomic_numbers[symbol]
        return self._atomic_numbers_cache[symbol]

    def _quote(self, path: str) -> str:
        """
        Quotes a path for LAMMPS script safety after validation.
        Uses caching to avoid redundant validation calls.
        """
        # Sanitize input path
        # Note: path must be string for lru_cache
        safe_path = validate_path_safe(Path(path))
        if not safe_path.exists() and not safe_path.parent.exists():
            msg = f"Path {safe_path} is invalid or has an invalid parent directory."
            raise ValueError(msg)

        # Ensure potential path contains only safe characters
        path_str = str(safe_path)
        import re
        if not re.match(r"^[a-zA-Z0-9_./-]+$", path_str):
            msg = f"Path contains invalid characters: {path_str}"
            raise ValueError(msg)

        # Use shlex.quote for shell safety
        return shlex.quote(path_str)

    from pyacemaker.core.lammps_template import ScriptTemplate

    def _gen_potential_pure(
        self, template: "ScriptTemplate", potential_path: Path, elements: list[str]
    ) -> None:
        """Generates pure PACE potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        template.write("pair_style pace\n")
        template.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

    def _gen_potential_hybrid(
        self, template: "ScriptTemplate", potential_path: Path, elements: list[str]
    ) -> None:
        """Generates hybrid PACE + ZBL potential commands."""
        species_str = " ".join(elements)
        quoted_pot = self._quote(str(potential_path))
        params = self.config.hybrid_params

        # Explicit type casting for security before injection
        inner = float(params.zbl_cut_inner)
        outer = float(params.zbl_cut_outer)

        template.write(
            f"pair_style hybrid/overlay pace zbl {inner} {outer}\n"
        )
        template.write(f"pair_coeff * * pace {quoted_pot} {species_str}\n")

        n_types = len(elements)

        # Optimize loop string concatenation
        # Use list comprehension for ZBL pairs
        zbl_lines = []
        for i in range(n_types):
            el_i = elements[i]
            z_i = self._get_atomic_number(el_i)
            for j in range(i, n_types):
                el_j = elements[j]
                z_j = self._get_atomic_number(el_j)
                zbl_lines.append(f"pair_coeff {i + 1} {j + 1} zbl {z_i} {z_j}\n")

        template.write("".join(zbl_lines))

    def _gen_potential(self, template: "ScriptTemplate", potential_path: Path, elements: list[str]) -> None:
        """Generates potential definition commands."""
        if self.config.hybrid_potential:
            self._gen_potential_hybrid(template, potential_path, elements)
        else:
            self._gen_potential_pure(template, potential_path, elements)

    def _gen_settings(self, template: "ScriptTemplate") -> None:
        """Generates general MD settings."""
        template.write(f"neighbor {self.config.neighbor_skin} bin\n")
        template.write("neigh_modify delay 0 every 1 check yes\n")
        template.write(f"timestep {self.config.timestep}\n")

    def _gen_watchdog(self, template: "ScriptTemplate", potential_path: Path, use_fix_invoke: bool = False, eval_dir: Path | None = None) -> None:
        """Generates Uncertainty Watchdog commands."""
        if not self.config.fix_halt:
            return

        quoted_pot = self._quote(str(potential_path))
        template.write(f"compute gamma all pace {quoted_pot}\n")
        template.write("compute max_gamma all reduce max c_gamma\n")
        template.write("variable max_g equal c_max_gamma\n")

        if use_fix_invoke and self.config.evaluator_thresholds and eval_dir is not None:
            # We add a boolean variable to hold the trigger from TwoTierEvaluator
            template.write("variable trigger_halt string false\n")

            # The parameters for TwoTierEvaluator
            # Validation handled by Pydantic domain model
            threshold_call = self.config.evaluator_thresholds.threshold_call_dft
            threshold_add = self.config.evaluator_thresholds.threshold_add_train
            smooth_steps = self.config.evaluator_thresholds.smooth_steps
            max_retries = self.config.evaluator_thresholds.max_retries
            base_backoff = self.config.evaluator_thresholds.base_backoff

            eval_dir.mkdir(parents=True, exist_ok=True)
            evaluator_script_path = eval_dir / "evaluator_script.py"

            with evaluator_script_path.open("w") as eval_f:
                eval_f.write("from pyacemaker.core.evaluator import TwoTierEvaluator\n")
                eval_f.write("import lammps\n")
                eval_f.write(f"evaluator = TwoTierEvaluator({threshold_call}, {threshold_add}, {smooth_steps}, {max_retries}, {base_backoff})\n")
                eval_f.write("def lammps_invoke_evaluator(*args, **kwargs):\n")
                eval_f.write("    lmp = lammps.lammps(name='', cmdargs=['-log', 'none', '-screen', 'none'])\n")
                eval_f.write("    evaluator.evaluate(lmp)\n")

            quoted_evaluator_script = self._quote(str(evaluator_script_path))
            # Use file-based python invocation rather than inline strings (Security requirement from memory)
            template.write(f"python invoke_evaluator invoke lammps_invoke_evaluator file {quoted_evaluator_script}\n")
            template.write(f"fix invoke_eval all python/invoke {self.config.check_interval} end_of_step invoke_evaluator\n")

            # Secondary halt check that relies on the trigger variable from TwoTierEvaluator
            template.write(
                f"fix halt_trigger all halt {self.config.check_interval} "
                f"v_trigger_halt == true error continue\n"
            )
        else:
            # The traditional halt check based on max_g
            template.write(
                f"fix halt_check all halt {self.config.check_interval} "
                f"v_max_g > {self.config.uncertainty_threshold} error continue\n"
            )

    def _gen_mc(self, template: "ScriptTemplate", elements: list[str]) -> None:
        """Generates Monte Carlo atom swapping commands."""
        if not self.config.mc:
            return

        n_types = len(elements)
        if n_types < 2:
            return  # Can't swap if fewer than 2 types

        # types keyword requires list of types to swap
        types_str = " ".join(str(i + 1) for i in range(n_types))

        # Command syntax: fix mc all atom/swap N X seed T types {types}
        # N: swap frequency (steps)
        # X: swaps per attempt (set to 1)
        # T: temperature (for Boltzmann factor)

        temp = self.config.temperature
        if self.config.ramping and self.config.ramping.temp_start is not None:
            temp = self.config.ramping.temp_start

        template.write(
            f"fix mc_swap all atom/swap {self.config.mc.swap_freq} 1 {self.config.mc.seed} "
            f"{temp} ke no types {types_str}\n"
        )

    # ruff: noqa: C901
    def _gen_execution(self, template: "ScriptTemplate", elements: list[str], resume_from_step: int | None = None, restart_file: Path | None = None) -> None:
        """Generates minimization and MD run commands."""
        if self.config.minimize and resume_from_step is None:
            template.write(
                f"minimize {self.config.minimize_tol} {self.config.minimize_ftol} "
                f"{self.config.minimize_steps} {self.config.minimize_max_iter}\n"
            )

        # MC
        self._gen_mc(template, elements)

        # Calculate damping parameters
        tdamp = self.config.tdamp_factor * self.config.timestep
        pdamp = self.config.pdamp_factor * self.config.timestep

        # Determine T/P start/end
        temp_start = self.config.temperature
        temp_end = self.config.temperature
        press_start = self.config.pressure
        press_end = self.config.pressure

        if self.config.ramping:
            if self.config.ramping.temp_start is not None:
                temp_start = self.config.ramping.temp_start
            if self.config.ramping.temp_end is not None:
                temp_end = self.config.ramping.temp_end
            if self.config.ramping.press_start is not None:
                press_start = self.config.ramping.press_start
            if self.config.ramping.press_end is not None:
                press_end = self.config.ramping.press_end

        if resume_from_step is None:
            # Use configurable velocity seed
            template.write(f"velocity all create {temp_start} {self.config.velocity_seed}\n")
            template.write(
                f"fix npt all npt temp {temp_start} {temp_end} {tdamp} "
                f"iso {press_start} {press_end} {pdamp}\n"
            )
        else:
            # When resuming, use a strong Langevin thermostat for the first few steps (soft start)
            if self.config.soft_start_steps > 0:
                template.write(
                    f"fix soft_start all langevin {temp_start} {temp_end} {tdamp} {self.config.velocity_seed}\n"
                )
                template.write("fix nve all nve\n")
                template.write(f"run {self.config.soft_start_steps}\n")
                template.write("unfix soft_start\n")
                template.write("unfix nve\n")

            template.write(
                f"fix npt all npt temp {temp_start} {temp_end} {tdamp} "
                f"iso {press_start} {press_end} {pdamp}\n"
            )

        # Write restart file logic
        if restart_file:
            quoted_restart = self._quote(str(restart_file))
            # Only write restart on regular interval if not using python halt
            template.write(f"restart 1000 {quoted_restart}\n")

        # How many steps left
        run_steps = self.config.n_steps
        if resume_from_step is not None:
            run_steps = max(0, self.config.n_steps - resume_from_step)
            # Subtract soft start steps from remaining run
            run_steps = max(0, run_steps - self.config.soft_start_steps)

        template.write(f"run {run_steps}\n")

        # Write final restart
        if restart_file:
            quoted_restart = self._quote(str(restart_file))
            template.write(f"write_restart {quoted_restart}\n")

    def _gen_output_setup(self, template: "ScriptTemplate", dump_file: Path) -> None:
        """Generates output settings (thermo and dump)."""
        template.write(f"thermo {self.config.thermo_freq}\n")

        style_parts = ["step", "temp", "pe", "press"]
        dump_parts = ["id", "type", "x", "y", "z"]

        if self.config.fix_halt:
            style_parts.append("v_max_g")
            dump_parts.append("c_gamma")

        style = " ".join(style_parts)
        dump_cols = " ".join(dump_parts)

        quoted_dump = self._quote(str(dump_file))
        template.write(f"thermo_style custom {style}\n")
        template.write(f"dump traj all custom {self.config.dump_freq} {quoted_dump} {dump_cols}\n")

        # Define variables for extraction via Python interface
        vars_to_export = ["pe", "temp", "step", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
        for v in vars_to_export:
            template.write(f"variable {v} equal {v}\n")

    def _gen_post_run_diagnostics(self, template: "ScriptTemplate") -> None:
        """Generates post-run diagnostic prints."""

    from pyacemaker.domain_models.md import ScriptGenerationContext

    def write_script(
        self,
        buffer: TextIO,
        ctx: "ScriptGenerationContext",
    ) -> None:
        """
        Writes the LAMMPS input script to the provided buffer.
        """
        from pyacemaker.core.lammps_template import ScriptTemplate
        template = ScriptTemplate(buffer)

        template.write("clear\n")

        if ctx.read_restart:
            quoted_read_restart = self._quote(str(ctx.read_restart))
            template.write(f"read_restart {quoted_read_restart}\n")
        else:
            template.write("units metal\n")
            # Use .value to ensure we get the string value "atomic", "charge" etc.
            template.write(f"atom_style {self.config.atom_style.value}\n")
            template.write("boundary p p p\n")
            quoted_data = self._quote(str(ctx.data_file))
            template.write(f"read_data {quoted_data}\n")

        self._gen_potential(template, ctx.potential_path, ctx.elements)
        self._gen_settings(template)
        self._gen_watchdog(template, ctx.potential_path, use_fix_invoke=ctx.use_fix_invoke, eval_dir=ctx.eval_dir)

        # Output setup MUST come before run
        self._gen_output_setup(template, ctx.dump_file)

        self._gen_execution(template, ctx.elements, resume_from_step=ctx.resume_from_step, restart_file=ctx.restart_file)

        self._gen_post_run_diagnostics(template)

    def write_minimization_script(
        self,
        buffer: TextIO,
        potential_path: Path,
        data_file: Path,
        elements: list[str],
    ) -> None:
        """
        Writes a minimization-only script for relaxation.
        """
        from pyacemaker.core.lammps_template import ScriptTemplate
        template = ScriptTemplate(buffer)

        quoted_data = self._quote(str(data_file))

        template.write("clear\n")
        template.write("units metal\n")
        template.write(f"atom_style {self.config.atom_style.value}\n")
        template.write("boundary p p p\n")
        template.write(f"read_data {quoted_data}\n")

        self._gen_potential(template, potential_path, elements)

        template.write(f"neighbor {self.config.neighbor_skin} bin\n")
        template.write("neigh_modify delay 0 every 1 check yes\n")
        template.write(f"min_style {LAMMPS_MIN_STYLE_CG}\n")
        template.write(
            f"minimize {self.config.minimize_tol} {self.config.minimize_ftol} "
            f"{self.config.minimize_steps} {self.config.minimize_max_iter}\n"
        )
