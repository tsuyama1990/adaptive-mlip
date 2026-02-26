from unittest.mock import patch

import pytest

from pyacemaker.domain_models.constants import DEFAULT_FALLBACK_CELL_SIZE
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig
from pyacemaker.structure_generator.direct import DirectSampler


@pytest.fixture
def structure_config():
    return StructureConfig(
        elements=["Cu"],
        supercell_size=[2, 2, 2],
        num_structures=1,
        active_policies=[ExplorationPolicy.COLD_START]
    )

def test_create_template_fallback(structure_config):
    """Test that _create_template falls back to cubic cell if bulk fails."""
    sampler = DirectSampler(structure_config)

    # Patch ase.build.bulk to raise an generic Exception
    # NOTE: SIM117 (nested with) is ignored here for readability/compatibility if needed,
    # but I will combine them to be clean as per previous linting hint.
    with patch("ase.build.bulk", side_effect=Exception("Bulk generation failed")), \
         patch("logging.getLogger"):

        template = sampler._create_template()

        # Should be cubic box of size DEFAULT_FALLBACK_CELL_SIZE * supercell
        # Primitive fallback is size [10, 10, 10].
        # Multiplied by [2, 2, 2] -> [20, 20, 20]

        cell = template.get_cell()
        # Ensure we are comparing float to float
        expected_len = float(DEFAULT_FALLBACK_CELL_SIZE * 2.0)

        # Check cell dimensions
        assert cell[0][0] == expected_len
        # Fallback creates 1 atom per primitive cell (Atoms('Cu', ...)).
        # Repeated 2x2x2 -> 8 atoms.
        assert len(template) == 8
