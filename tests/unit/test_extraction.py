from ase.build import bulk

from pyacemaker.domain_models.workflow import CutoutConfig
from pyacemaker.utils.extraction import extract_intelligent_cluster


def test_extract_intelligent_cluster():
    structure = bulk("Cu", "fcc", a=3.6)
    structure = structure * (3, 3, 3)  # 27 atoms

    config = CutoutConfig(core_radius=2.0, buffer_radius=2.0)

    # Target central atom (roughly atom index 13)
    target_atoms = [13]

    cluster = extract_intelligent_cluster(structure, target_atoms, config)

    assert len(cluster) > 1  # At least the target atom and some neighbors

    # Check force_weight array exists
    weights = cluster.get_array("force_weight")
    assert weights is not None
    assert weights.shape[0] == len(cluster)
    assert sum(weights) >= 1.0  # At least the target atom should have weight 1.0
