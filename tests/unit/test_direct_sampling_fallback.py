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

    # Patch the 'bulk' function imported in the direct module, NOT the original ase.build.bulk
    with patch("pyacemaker.structure_generator.direct.bulk", side_effect=Exception("Bulk generation failed")), \
         patch("logging.getLogger"):

        template = sampler._create_template()

        # Should be cubic box of size DEFAULT_FALLBACK_CELL_SIZE * supercell
        # Primitive fallback is size [10, 10, 10].
        # Multiplied by [2, 2, 2] -> [20, 20, 20]

        cell = template.get_cell()
        expected_len = float(DEFAULT_FALLBACK_CELL_SIZE * 2.0)

        # Check cell dimensions
        # Debug info: print cell if assertion fails
        print(f"DEBUG: cell={cell}, expected={expected_len}")

        # We expect a diagonal cell [20, 20, 20]
        assert cell[0][0] == expected_len
        # Fallback creates 1 atom per primitive cell (Atoms('Cu', ...)).
        # Repeated 2x2x2 -> 8 atoms.
        assert len(template) == 8
