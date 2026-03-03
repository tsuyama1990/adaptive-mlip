from ase import Atoms

from pyacemaker.domain_models.defaults import ERR_M3GNET_PRED_FAIL


class M3GNetWrapper:
    """
    Wrapper for M3GNet structure prediction.
    Currently uses a mock implementation (ase.build.bulk) for 'cold start'.
    """

    def __init__(self, use_mock: bool = False) -> None:
        """
        Initialize the M3GNet wrapper.

        Args:
            use_mock: If True, uses a mock structure generator (for testing only).
        """
        self.use_mock = use_mock

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
        if self.use_mock:
            return self._mock_predict(composition)

        try:
            # Actual M3GNet import and usage
            from pymatgen.ext.matproj import MPRester
            from pymatgen.io.ase import AseAtomsAdaptor

            def _raise_error() -> None:
                msg = f"No structures found for {composition} in Materials Project."
                raise ValueError(msg)  # noqa: TRY301

            def _predict() -> Atoms:
                import time
                import logging
                logger = logging.getLogger(__name__)

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with MPRester() as mpr:
                            # Fetch structures from Materials Project
                            docs = mpr.materials.search(formula=composition, is_stable=True)
                            if not docs:
                                docs = mpr.materials.search(formula=composition)

                        if not docs:
                            _raise_error()

                        # Get the most stable one
                        structure = docs[0].structure
                        return AseAtomsAdaptor.get_atoms(structure)  # type: ignore[no-any-return]
                    except Exception as loop_e:
                        if attempt == max_retries - 1:
                            raise loop_e
                        backoff = 2 ** attempt
                        logger.warning(f"MPRester network failure, retrying in {backoff}s: {loop_e!s}")
                        time.sleep(backoff)

                _raise_error()
                return Atoms()

            return _predict()

        except Exception as e:
            raise RuntimeError(ERR_M3GNET_PRED_FAIL.format(composition=composition)) from e

    def _mock_predict(self, composition: str) -> Atoms:
        from ase.build import bulk

        # Simple Mock logic
        if composition == "FePt":
            return Atoms(
                "FePt",
                positions=[[0, 0, 0], [1.9, 1.9, 1.9]],
                cell=[3.8, 3.8, 3.8],
                pbc=True,
            )

        # Fallback to bulk or simple cubic
        try:
            return bulk(composition)
        except Exception:
            # Very simple fallback
            return Atoms(composition, cell=[5.0, 5.0, 5.0], pbc=True)
