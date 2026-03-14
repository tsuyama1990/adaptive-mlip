from typing import cast

import networkx as nx
from ase.data import atomic_masses, chemical_symbols

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.dft import DFTConfig
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
    def compile(cls, intent: IntentRequest) -> PyAceConfig:  # noqa: C901
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
                    structure_config = cls._compile_initial_structure_node(node, slider, material)
                case NodeType.MACE_TRAINING:
                    training_config = cls._compile_mace_training_node(node, slider, material)
                case NodeType.ACTIVE_LEARNING_LOOP:
                    md_config, dft_config, workflow_config = cls._compile_active_learning_node(
                        node, slider, material
                    )
                case _:
                    msg = f"Unsupported node type: {node.type}"
                    raise NotImplementedError(msg)

        # Inject intelligent defaults for any missing components
        if structure_config is None:
            # Although logical validation catches missing struct before loop,
            # this ensures PyAceConfig is fully populated
            structure_config = cls._compile_initial_structure_node(
                DagNode(
                    id="default",
                    type=NodeType.INITIAL_STRUCTURE,
                    data=InitialStructureData(
                        type=NodeType.INITIAL_STRUCTURE,
                        chemical_symbol=material,
                        lattice_constant=4.0,
                    ),
                ),
                slider,
                material,
            )

        if training_config is None:
            training_config = cls._compile_mace_training_node(None, slider, material)

        if md_config is None or dft_config is None or workflow_config is None:
            md, dft, wf = cls._compile_active_learning_node(None, slider, material)
            if md_config is None:
                md_config = md
            if dft_config is None:
                dft_config = dft
            if workflow_config is None:
                workflow_config = wf

        return PyAceConfig(
            project_name="intent_driven_project",
            structure=structure_config,
            dft=dft_config,
            training=training_config,
            md=md_config,
            workflow=workflow_config,
            eon=None,
            scenario=None,
        )

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

        sorted_ids = list(nx.topological_sort(graph))
        return [nodes_dict[nid] for nid in sorted_ids]

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
        slider: int,
        material: str,  # noqa: ARG003
    ) -> StructureConfig:
        data = cast(InitialStructureData, node.data)

        return StructureConfig(elements=[data.chemical_symbol], supercell_size=[3, 3, 3])

    @classmethod
    def _compile_mace_training_node(
        cls,
        node: DagNode | None,
        slider: int,
        material: str,  # noqa: ARG003
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
    def _compile_active_learning_node(
        cls,
        node: DagNode | None,
        slider: int,
        material: str,  # noqa: ARG003
    ) -> tuple[MDConfig, DFTConfig, WorkflowConfig]:
        # Intelligent defaults mapping based on material and slider

        # 1. Determine base mass and timestep
        try:
            z = chemical_symbols.index(material)
            mass = atomic_masses[z]
        except ValueError:
            mass = 1.0  # default

        # Lighter elements -> smaller timestep.
        timestep = 1.0
        if mass > 50.0:
            timestep = 2.0
        if mass < 10.0:
            timestep = 0.5

        # Scale thresholds based on accuracy slider (1=Speed, 10=Accuracy)
        # Accuracy=10 -> lower thresholds (stricter)
        conv_thr = 1e-4 * (11 - slider) / 10.0  # e.g., slider=10 -> 1e-5
        conv_thr = max(conv_thr, 1e-6)

        # Higher slider -> more accurate -> lower threshold to call DFT
        call_dft_thr = 0.01 + 0.1 * ((10 - slider) / 9.0)  # slider=10 -> 0.01, slider=1 -> 0.11

        md_config = MDConfig(
            temperature=300.0,
            pressure=1.0,
            timestep=timestep,
            n_steps=1000,
            units="metal",
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
        )

        dft_config = DFTConfig(
            code="quantum_espresso",
            functional="pbe",
            kpoints_density=2.0 + (10 - slider) * 0.5,  # adjust density based on slider
            encut=40.0 + (slider * 2.0),  # e.g. slider 10 -> 60 eV
            pseudopotentials={material: f"{material}.pbe-n-kjpaw_psl.1.0.0.UPF"},
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
                    threshold_call_dft=call_dft_thr, threshold_add_train=call_dft_thr / 2.0
                )
            ),
        )

        return md_config, dft_config, workflow_config
