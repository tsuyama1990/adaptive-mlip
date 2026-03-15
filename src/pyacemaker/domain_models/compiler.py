import concurrent.futures
from typing import cast

import networkx as nx
from ase.data import atomic_masses, chemical_symbols

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.dft import DFTConfig
from pyacemaker.domain_models.gui_schema import SpatialAction
from pyacemaker.domain_models.md import AtomStyle, MDConfig
from pyacemaker.domain_models.scenario import (
    DagNode,
    InitialStructureData,
    IntentRequest,
    NodeType,
)
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.domain_models.training import PacemakerConfig, TrainingConfig
from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    LoopStrategyConfig,
    WorkflowConfig,
)


class SemanticCompiler:
    @classmethod
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

        for node in sorted_nodes:
            match node.type:
                case NodeType.INITIAL_STRUCTURE:
                    structure_config = cls._compile_initial_structure_node(node)
                case NodeType.MACE_TRAINING:
                    training_config = cls._compile_mace_training_node(material)
                case NodeType.ACTIVE_LEARNING_LOOP:
                    # Find the structure node if it exists
                    structure_node = next(
                        (n for n in sorted_nodes if n.type == NodeType.INITIAL_STRUCTURE), None
                    )
                    md_config, dft_config, workflow_config = cls._compile_active_learning_node(
                        slider, material, structure_node
                    )
                case _:
                    msg = f"Unsupported node type: {node.type}"
                    raise NotImplementedError(msg)

        if structure_config is None:
            msg = "Missing INITIAL_STRUCTURE node in workflow"
            raise CompilerError(msg)

        if training_config is None:
            msg = "Missing MACE_TRAINING node in workflow"
            raise CompilerError(msg)

        if md_config is None or dft_config is None or workflow_config is None:
            msg = "Missing ACTIVE_LEARNING_LOOP node in workflow"
            raise CompilerError(msg)

        from pydantic import ValidationError

        from pyacemaker.domain_models.defaults import DEFAULT_PROJECT_NAME

        try:
            return PyAceConfig(
                project_name=DEFAULT_PROJECT_NAME,
                structure=structure_config,
                dft=dft_config,
                training=training_config,
                md=md_config,
                workflow=workflow_config,
            )  # type: ignore[call-arg]
        except ValidationError as e:
            msg = f"Final compilation validation failed: {e!s}"
            raise CompilerError(msg) from e

    @classmethod
    def _topological_sort(cls, intent: IntentRequest) -> list[DagNode]:
        graph = nx.DiGraph()
        nodes_dict = {node.id: node for node in intent.nodes}

        for node in intent.nodes:
            graph.add_node(node.id)

        for edge in intent.edges:
            graph.add_edge(edge.source, edge.target)

        if not nx.is_directed_acyclic_graph(graph):
            msg = "Cycle detected in DAG."
            raise CompilerError(msg)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda g: list(nx.topological_sort(g)), graph)
                sorted_ids = future.result(timeout=10.0)
            return [nodes_dict[nid] for nid in sorted_ids]
        except concurrent.futures.TimeoutError as err:
            msg = "DAG processing timeout"
            raise CompilerError(msg) from err

    @classmethod
    def _validate_sequence(cls, sorted_nodes: list[DagNode]) -> None:
        has_struct = False

        for node in sorted_nodes:
            if node.type == NodeType.INITIAL_STRUCTURE:
                has_struct = True
            elif (
                node.type
                in (
                    NodeType.ACTIVE_LEARNING_LOOP,
                    NodeType.MACE_TRAINING,
                    NodeType.EON_TRANSITION_SEARCH,
                )
                and not has_struct
            ):
                msg = "INITIAL_STRUCTURE node must precede any training or execution nodes in the workflow sequence."
                raise CompilerError(msg)

        # Check for branching parallel active learning loops.
        # This is a bit simplistic: if there are multiple ACTIVE_LEARNING nodes, we reject it.
        # The true check would be looking at the graph out-degree of structure nodes.
        active_loop_count = sum(1 for n in sorted_nodes if n.type == NodeType.ACTIVE_LEARNING_LOOP)
        if active_loop_count > 1:
            msg = "Parallel active learning loops originating from a single structure are currently not supported by the orchestrator."
            raise CompilerError(msg)

    @classmethod
    def _compile_initial_structure_node(
        cls,
        node: DagNode,
    ) -> StructureConfig:
        data = cast(InitialStructureData, node.data)

        return StructureConfig(elements=[data.chemical_symbol], supercell_size=[3, 3, 3])

    @classmethod
    def _compile_mace_training_node(
        cls,
        material: str,
    ) -> TrainingConfig:
        # Intelligent defaults
        return TrainingConfig(
            potential_type="mace",
            cutoff_radius=5.0,
            max_basis_size=8,
            pacemaker=PacemakerConfig(),
            elements=[material],
            delta_learning=False,
            active_set_optimization=False,
            active_set_size=None,
            max_iterations=1000,
            batch_size=8,
        )

    @classmethod
    def _compile_active_learning_node(  # noqa: C901
        cls,
        slider: int,
        material: str,
        structure_node: DagNode | None = None,
    ) -> tuple[MDConfig, DFTConfig, WorkflowConfig]:
        # Intelligent defaults mapping based on material and slider

        # 1. Determine base mass and timestep
        import re

        if not re.match(r"^[A-Z][a-z]?$", material) or material not in chemical_symbols:
            msg = "Invalid chemical symbol"
            raise CompilerError(msg)

        if not (1 <= slider <= 10):
            msg = "Slider must be between 1 and 10"
            raise CompilerError(msg)

        z = chemical_symbols.index(material)
        mass = atomic_masses[z]

        # Lighter elements -> smaller timestep.
        timestep = 1.0
        if mass > 50.0:
            timestep = 2.0
        if mass < 10.0:
            timestep = 0.5

        # Scale thresholds based on accuracy slider (1=Speed, 10=Accuracy)
        # Accuracy=10 -> lower thresholds (stricter)
        conv_thr = 1e-5 + (1e-4 - 1e-5) * (10 - slider) / 9.0  # e.g., slider=10 -> 1e-5
        conv_thr = max(conv_thr, 1e-6)

        # Higher slider -> more accurate -> lower threshold to call DFT
        call_dft_thr = 0.01 + 0.1 * (10 - slider) / 9.0  # slider=10 -> 0.01, slider=1 -> 0.11

        from pyacemaker.domain_models.defaults import (
            DEFAULT_MD_PRESSURE,
            DEFAULT_MD_TEMPERATURE,
            DEFAULT_MD_UNITS,
        )

        spatial_tags_commands = None
        ignore_tags = None
        if structure_node is not None:
            data = cast(InitialStructureData, structure_node.data)
            if data.spatial_regions:
                spatial_tags_commands = []
                ignore_tags = []
                # Map bounding boxes to LAMMPS logic
                for i, region in enumerate(data.spatial_regions, start=1):
                    # For a region we define a block and group
                    cmd_region = f"region reg_{i} block {region.x_min} {region.x_max} {region.y_min} {region.y_max} {region.z_min} {region.z_max}"
                    cmd_group = f"group group_{i} region reg_{i}"
                    spatial_tags_commands.extend([cmd_region, cmd_group])

                    if region.action == SpatialAction.ACTION_FREEZE:
                        cmd_fix = f"fix fix_{i} group_{i} setforce 0.0 0.0 0.0"
                        spatial_tags_commands.append(cmd_fix)
                        ignore_tags.append(1)  # 1 is the tag mapped to FREEZE in spatial module
                    elif region.action == SpatialAction.ACTION_LANGEVIN_THERMOSTAT:
                        cmd_fix = f"fix fix_{i} group_{i} langevin {DEFAULT_MD_TEMPERATURE} {DEFAULT_MD_TEMPERATURE} 0.1 48279"
                        spatial_tags_commands.append(cmd_fix)

        md_config = MDConfig(
            temperature=DEFAULT_MD_TEMPERATURE,
            pressure=DEFAULT_MD_PRESSURE,
            timestep=timestep,
            n_steps=1000,
            units=DEFAULT_MD_UNITS,
            atom_style=AtomStyle.ATOMIC,
            thermo_freq=100,
            dump_freq=100,
            minimize=False,
            neighbor_skin=2.0,
            tdamp_factor=100.0,
            pdamp_factor=1000.0,
            hybrid_potential=False,
            fix_halt=True,
            uncertainty_threshold=call_dft_thr,
            check_interval=10,
            ramping=None,
            mc=None,
            soft_start_steps=0,
            soft_start_langevin_damp=0.1,
            spatial_tags_commands=spatial_tags_commands,
        )

        from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_CODE,
            DEFAULT_DFT_FUNCTIONAL,
            DEFAULT_PSEUDOPOTENTIAL_MAPPING,
        )

        safe_pseudo = DEFAULT_PSEUDOPOTENTIAL_MAPPING.get(material)
        if not safe_pseudo:
            msg = f"No verified pseudopotential mapping exists for material: {material}"
            raise CompilerError(msg)

        dft_config = DFTConfig(
            code=DEFAULT_DFT_CODE,
            functional=DEFAULT_DFT_FUNCTIONAL,
            kpoints_density=2.0 + 4.0 * (10 - slider) / 9.0,  # adjust density based on slider
            encut=40.0 + (slider * 2.0),  # e.g. slider 10 -> 60 eV
            pseudopotentials={material: safe_pseudo},
            embedding_buffer=None,
            mixing_beta=0.7,
            smearing_type="mv",
            smearing_width=0.01,
            diagonalization="david",
            mixing_beta_factor=0.5,
            smearing_width_factor=2.0,
        )

        workflow_config = WorkflowConfig(
            max_iterations=10,
            loop_strategy=LoopStrategyConfig(
                thresholds=ActiveLearningThresholds(
                    threshold_call_dft=call_dft_thr,
                    threshold_add_train=call_dft_thr / 2.0,
                    ignore_tags=ignore_tags,
                )
            ),
        )

        return md_config, dft_config, workflow_config
