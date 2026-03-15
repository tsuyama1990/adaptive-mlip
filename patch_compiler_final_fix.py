import re

with open("src/pyacemaker/domain_models/compiler.py", "r") as f:
    content = f.read()

# I see the changes I made to compiler.py earlier failed entirely because they had syntax errors and were reverted, or I never actually applied them correctly because `ruff` fixes failed or something! Let's manually apply the active learning node regions logic!

replacement_compile = """    @classmethod
    def compile(cls, intent: IntentRequest) -> PyAceConfig:
        sorted_nodes = cls._topological_sort(intent)
        cls._validate_sequence(sorted_nodes)

        # Build components
        structure_config = None
        training_config = None
        md_config = None
        dft_config = None
        workflow_config = None

        slider = intent.accuracy_speed_slider
        material = intent.target_material
        regions = None

        for node in sorted_nodes:
            match node.type:
                case NodeType.INITIAL_STRUCTURE:
                    structure_config, regions = cls._compile_initial_structure_node(node)
                case NodeType.MACE_TRAINING:
                    training_config = cls._compile_mace_training_node(material)
                case NodeType.ACTIVE_LEARNING_LOOP:
                    md_config, dft_config, workflow_config = cls._compile_active_learning_node(
                        slider, material, regions
                    )
                case _:
                    msg = f"Unsupported node type: {node.type}"
                    raise NotImplementedError(msg)"""

content = re.sub(
    r"    @classmethod\n    def compile\(cls, intent: IntentRequest\) -> PyAceConfig:[\s\S]*?raise NotImplementedError\(msg\)",
    replacement_compile,
    content
)

replacement_initial_structure = """    @classmethod
    def _compile_initial_structure_node(
        cls,
        node: DagNode,
    ) -> tuple[StructureConfig, list]:
        data = cast(InitialStructureData, node.data)

        from pyacemaker.domain_models.gui_schema import SpatialRegion
        regions: list[SpatialRegion] = data.regions if data.regions else []

        return StructureConfig(elements=[data.chemical_symbol], supercell_size=[3, 3, 3]), regions"""

content = re.sub(
    r"    @classmethod\n    def _compile_initial_structure_node\([\s\S]*?return StructureConfig\(elements=\[data\.chemical_symbol\], supercell_size=\[3, 3, 3\]\)",
    replacement_initial_structure,
    content
)

replacement_active_learning = """    @classmethod
    def _compile_active_learning_node(
        cls,
        slider: int,
        material: str,
        regions: list,
    ) -> tuple[MDConfig, DFTConfig, WorkflowConfig]:"""

content = re.sub(
    r"    @classmethod\n    def _compile_active_learning_node\([\s\S]*?\) -> tuple\[MDConfig, DFTConfig, WorkflowConfig\]:",
    replacement_active_learning,
    content
)

logic_replacement = """        workflow_config = WorkflowConfig(
            max_iterations=10,
            loop_strategy=LoopStrategyConfig(
                thresholds=ActiveLearningThresholds(
                    threshold_call_dft=call_dft_thr, threshold_add_train=call_dft_thr / 2.0
                )
            ),
        )

        if regions:
            import numpy as np
            from ase.build import bulk
            from pyacemaker.utils.spatial import apply_spatial_tags
            from pyacemaker.domain_models.gui_schema import PhysicalAction

            atoms = bulk(material, "fcc", a=3.0).repeat((3, 3, 3))
            tags = apply_spatial_tags(atoms, regions)

            cmds = []
            freeze_indices = []

            for i, region in enumerate(regions):
                region_name = f"reg_{i+1}"
                cmds.append(f"region {region_name} block {region.x_min} {region.x_max} {region.y_min} {region.y_max} {region.z_min} {region.z_max} units box")

            for i, region in enumerate(regions):
                tag_value = i + 1
                region_name = f"reg_{tag_value}"
                group_name = f"group_{tag_value}"

                cmds.append(f"group {group_name} region {region_name}")

            for i, region in enumerate(regions):
                tag_value = i + 1
                group_name = f"group_{tag_value}"

                for j, other_region in enumerate(regions):
                    if i == j:
                        continue

                    priority_map = {
                        PhysicalAction.ACTION_ACTIVE_LEARNING_ONLY: 1,
                        PhysicalAction.ACTION_LANGEVIN_THERMOSTAT: 2,
                        PhysicalAction.ACTION_FREEZE: 3,
                    }
                    if priority_map[other_region.action] > priority_map[region.action]:
                        other_group = f"group_{j+1}"
                        cmds.append(f"group {group_name} subtract {group_name} {other_group}")

            for i, region in enumerate(regions):
                tag_value = i + 1
                group_name = f"group_{tag_value}"

                if region.action == PhysicalAction.ACTION_FREEZE:
                    cmds.append(f"fix freeze_fix_{tag_value} {group_name} setforce 0.0 0.0 0.0")
                    cmds.append(f"group freeze_atoms union {group_name}")
                    indices = np.where(tags == tag_value)[0] + 1
                    freeze_indices.extend(indices.tolist())

            if freeze_indices:
                cmds.append("group active_atoms subtract all freeze_atoms")
            else:
                cmds.append("group active_atoms union all")

            md_config.custom_initialization_commands = cmds
            workflow_config.loop_strategy.thresholds.ignored_atoms = list(set(freeze_indices))

        return md_config, dft_config, workflow_config"""

content = re.sub(
    r"        workflow_config = WorkflowConfig\([\s\S]*?return md_config, dft_config, workflow_config",
    logic_replacement,
    content
)

with open("src/pyacemaker/domain_models/compiler.py", "w") as f:
    f.write(content)
