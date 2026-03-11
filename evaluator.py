
def invoke_evaluator() -> None:
    import lammps
    lmp = lammps.lammps()

    try:
        max_g = lmp.extract_variable("max_g", 0, 0)
        threshold = lmp.extract_variable("threshold_dft", 0, 0)
        smooth_steps = lmp.extract_variable("smooth_steps", 0, 0)

        # Try to get consecutive_exceed, default to 0
        try:
            exceed = lmp.extract_variable("consecutive_exceed", 0, 0)
        except Exception:
            exceed = 0.0

        if max_g > threshold:
            exceed += 1.0
        else:
            exceed = 0.0

        lmp.command(f"variable consecutive_exceed equal {exceed}")
    except Exception:
        import traceback
        traceback.print_exc()
