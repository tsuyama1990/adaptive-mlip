def invoke_evaluator() -> None:
    import traceback
    try:
        import lammps
    except ImportError:
        # Avoid crashing if LAMMPS module not present during tests or parsing
        return

    lmp = lammps.lammps()

    try:
        max_g = lmp.extract_variable("max_g", 0, 0)
        threshold = lmp.extract_variable("threshold_dft", 0, 0)

        # Try to get consecutive_exceed, default to 0
        try:
            exceed = lmp.extract_variable("consecutive_exceed", 0, 0)
        except Exception:
            exceed = 0.0

        if max_g > threshold:
            exceed += 1.0
        else:
            exceed = 0.0

        # Ensure exceed is strictly a float before injection
        safe_exceed = float(exceed)

        import re

        from pyacemaker.domain_models.constants import MALICIOUS_SHELL_PATTERN

        # Safe assignment if no injection pattern found
        if not re.search(MALICIOUS_SHELL_PATTERN, str(safe_exceed)):
            lmp.command(f"variable consecutive_exceed equal {safe_exceed}")
    except Exception:
        traceback.print_exc()
