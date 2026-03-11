from ase import Atoms
from ase.build import bulk

from pyacemaker.domain_models.constants import ERR_M3GNET_PRED_FAIL


class M3GNetWrapper:
    """
    Wrapper for structure prediction.
    Uses ase.build.bulk for 'cold start'.
    """

    def predict_structure(self, composition: str) -> Atoms:
        """
        Predict a stable structure for the given composition.
        Args:
            composition: Chemical formula (e.g., 'Fe', 'NaCl').
        Returns:
            Atoms object.
        Raises:
            RuntimeError: If prediction fails after retries.
        """
        import re

        if not re.match(r"^[A-Za-z0-9]+$", composition):
            msg = f"Invalid composition string format: {composition}"
            raise ValueError(msg)

        try:
            return bulk(composition)
        except Exception as e:
            # Very simple fallback
            try:
                return Atoms(composition, cell=[5.0, 5.0, 5.0], pbc=True)
            except Exception:
                raise RuntimeError(ERR_M3GNET_PRED_FAIL.format(composition=composition)) from e
