with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

target = """        try:
            # Safely extract c_max_gamma
            # lmp is the LAMMPS python object instance
            max_gamma = float(lmp.extract_variable("max_g", None, 0))
        except Exception:
            logger.exception("Failed to extract max_g in evaluator")
            raise"""

custom = """        try:
            # Safely extract c_max_gamma
            # lmp is the LAMMPS python object instance
            max_gamma = float(lmp.extract_variable("max_g", None, 0))

            # Since we could not inject a group conditionally in python without syntax errors,
            # we will handle `ignored_atoms` by extracting the per-atom array directly if needed.
            # But the max_g variable is extracted from LAMMPS efficiently.
            # If ignored_atoms is populated, we can evaluate it from the LAMMPS array.
            if self.thresholds.ignored_atoms:
                import ctypes
                import numpy as np
                # The pace gamma is in c_gamma (per-atom). If we need to filter, we have to access the array.
                # It's safer to just extract `c_gamma` vector:
                try:
                    nlocal = lmp.extract_global("nlocal", 0)
                    gamma_array_ptr = lmp.extract_compute("gamma", 1, 1) # per-atom, array
                    if gamma_array_ptr:
                        # Convert to numpy array safely
                        gamma_data = np.ctypeslib.as_array(ctypes.cast(gamma_array_ptr, ctypes.POINTER(ctypes.c_double)), shape=(nlocal,))

                        # Get atom IDs to filter
                        atom_ids_ptr = lmp.extract_atom("id", 0)
                        atom_ids = np.ctypeslib.as_array(ctypes.cast(atom_ids_ptr, ctypes.POINTER(ctypes.c_int)), shape=(nlocal,))

                        # Mask out ignored atoms
                        ignored_set = set(self.thresholds.ignored_atoms)
                        mask = np.array([id_val not in ignored_set for id_val in atom_ids])

                        if np.any(mask):
                            max_gamma = float(np.max(gamma_data[mask]))
                        else:
                            max_gamma = 0.0
                except Exception as e:
                    logger.warning(f"Could not extract per-atom gamma to filter ignored atoms: {e}. Using global max.")
        except Exception:
            logger.exception("Failed to extract max_g in evaluator")
            raise"""

content = content.replace(target, custom)

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
