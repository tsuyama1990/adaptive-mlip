def invoke_evaluator():
    from lammps import lammps
    lmp = lammps(ptr=lammps.get_lammps_ptr())
    try:
        max_g = float(lmp.extract_variable('max_g', 0, 0))
        threshold = float(lmp.extract_variable('threshold_call_dft', 0, 0))
        smooth_steps = int(lmp.extract_variable('smooth_steps', 0, 0))
    except Exception as e:
        print(f'Evaluator variable extraction failed: {e}')
        return

    # Use LAMMPS internal variable for state instead of python sys.globals
    try:
        exceed_count = int(lmp.extract_variable('exceed_count', 0, 0))
    except Exception:
        exceed_count = 0

    if max_g > threshold:
        exceed_count += 1
        lmp.command(f'variable exceed_count equal {exceed_count}')
        if exceed_count >= smooth_steps:
            print('True Anomaly Detected, Halting')
            lmp.command('quit')
        else:
            print('Thermal Noise Detected, Ignoring')
    else:
        lmp.command('variable exceed_count equal 0')
