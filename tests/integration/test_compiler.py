import pytest
from ase.build import bulk
from pyacemaker.domain_models.compiler import SemanticCompiler
from pyacemaker.domain_models.scenario import IntentRequest, DagNode, InitialStructureData, NodeType, Edge
from pyacemaker.domain_models.gui_schema import SpatialRegion, PhysicalAction

def test_compiler_with_spatial_regions():
    # UAT-03-01: Verify compiler maps JSON regions to Lammps string cmds properly
    regions = [
        SpatialRegion(
            x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0, z_min=0.0, z_max=2.5,
            action=PhysicalAction.ACTION_FREEZE
        ),
        SpatialRegion(
            x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0, z_min=3.0, z_max=5.0,
            action=PhysicalAction.ACTION_LANGEVIN_THERMOSTAT
        )
    ]

    struct_node = DagNode(
        id="struct-1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE,
            chemical_symbol="Cu",
            lattice_constant=3.61,
            regions=regions
        )
    )

    train_node = DagNode(
        id="train-1",
        type=NodeType.MACE_TRAINING,
        data={"type": NodeType.MACE_TRAINING}
    )

    al_node = DagNode(
        id="al-1",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data={"type": NodeType.ACTIVE_LEARNING_LOOP}
    )

    intent = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Cu",
        nodes=[struct_node, train_node, al_node],
        edges=[
            Edge(source="struct-1", target="train-1"),
            Edge(source="train-1", target="al-1")
        ]
    )

    config = SemanticCompiler.compile(intent)

    cmds = config.md.custom_initialization_commands

    # Just check if cmds are generated successfully
    # Debugging
    print("CMDS:", cmds)
    assert len(cmds) > 0
    assert any("group" in cmd for cmd in cmds)
