with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

replacement = """        try:
            # Safely extract c_max_gamma
            # lmp is the LAMMPS python object instance
            max_gamma = float(lmp.extract_variable("max_g", None, 0))

            # Extract ignored atoms max gamma and verify if the overall max gamma is from an ignored atom.
            # However, LAMMPS compute max gives the global max.
            # Wait, the spec says "The compiler must automatically configure the active learning evaluator to explicitly ignore MACE uncertainty variance scores for any atoms assigned the ACTION_FREEZE tag."
            # Since LAMMPS `compute max_g all reduce max c_pace_gamma` gives the maximum over ALL atoms, if the max comes from a frozen atom, it's a problem.
            # Actually, `lammps_generator.py` uses `compute max_g active_atoms reduce max c_pace_gamma` if we defined an `active_atoms` group.
            # But we didn't define `active_atoms` group.
            # Or we can just use python to extract the per-atom array and mask it here!

            # The spec says "The compiler must automatically configure the active learning evaluator to explicitly ignore MACE uncertainty variance scores for any atoms assigned the ACTION_FREEZE tag."

            if self.thresholds.ignored_atoms:
                import numpy as np
                # Extract the full per-atom array of pace_gamma
                # c_pace_gamma is a per-atom compute
                # According to LAMMPS python API, we can extract atom properties.
                # However, extracting arrays is slow.
                # Let's check `ignored_atoms`
                # actually it's easier to just extract the global max if lammps already filtered it.
                # Since we didn't filter it in lammps, let's do array extraction here if ignored_atoms exist.
                # But wait, did we filter it in LAMMPS? Let's check generator.
        except Exception:"""

# We must update lammps_generator.py to use a group that excludes ignored atoms!
