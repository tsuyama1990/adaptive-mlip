import numpy as np


class MockMACEManager:
    def __init__(
        self,
        model_path: str = "dummy_model",
        c_gamma_val: float | tuple[float, float] = (0.01, 0.1),
        energy_val: float = -10.0,
    ) -> None:
        self.model_path = model_path
        self.c_gamma_val = c_gamma_val
        self.energy_val = energy_val

    def compute(self, structures, batch_size=10):
        for atoms in structures:
            atoms_copy = atoms.copy()
            atoms_copy.info["energy"] = self.energy_val

            if "forces" not in atoms_copy.arrays:
                atoms_copy.new_array("forces", np.zeros((len(atoms_copy), 3)))
            else:
                atoms_copy.set_array("forces", np.zeros((len(atoms_copy), 3)))

            if isinstance(self.c_gamma_val, tuple):
                c_gamma = np.random.uniform(
                    self.c_gamma_val[0], self.c_gamma_val[1], size=len(atoms_copy)
                )
            else:
                c_gamma = np.full(len(atoms_copy), self.c_gamma_val)

            if "c_gamma" not in atoms_copy.arrays:
                atoms_copy.new_array("c_gamma", c_gamma)
            else:
                atoms_copy.set_array("c_gamma", c_gamma)

            yield atoms_copy
