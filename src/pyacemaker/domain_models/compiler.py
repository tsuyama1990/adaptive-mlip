from typing import cast

import ase.build
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
    SpatialAction,
)
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.domain_models.training import PacemakerConfig, TrainingConfig
from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    LoopStrategyConfig,
    WorkflowConfig,
)
from pyacemaker.utils.spatial import apply_spatial_tags


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

        spatial_commands: list[str] | None = None

        for node in sorted_nodes:
            match node.type:
                case NodeType.INITIAL_STRUCTURE:
                    structure_config, spatial_commands = cls._compile_initial_structure_node(node)
                case NodeType.MACE_TRAINING:
                    training_config = cls._compile_mace_training_node(material)
                case NodeType.ACTIVE_LEARNING_LOOP:
                    md_config, dft_config, workflow_config = cls._compile_active_learning_node(
                        slider, material, spatial_commands
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
                eon=None,
                scenario=None,
            )
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
    ) -> tuple[StructureConfig, list[str] | None]:
        from pyacemaker.domain_models.defaults import DEFAULT_SUPERCELL_SIZE

        data = cast(InitialStructureData, node.data)

        spatial_commands: list[str] | None = None
        if data.regions:
            try:
                atoms = (
                    ase.build.bulk(data.chemical_symbol, a=data.lattice_constant, cubic=True)
                    * DEFAULT_SUPERCELL_SIZE
                )
            except Exception as e:
                msg = f"Failed to build initial structure for spatial tagging: {e}"
                raise CompilerError(msg) from e

            tags = apply_spatial_tags(atoms, data.regions)
            spatial_commands = []

            # Group by tag ID to avoid duplicate group definitions and handle 'all' optimization
            # Note: tags array contains values corresponding to region indices + 1, where 0 is untagged.
            import numpy as np

            unique_tags = np.unique(tags)

            for region_idx, region in enumerate(data.regions):
                tag_id = region_idx + 1
                if tag_id not in unique_tags:
                    continue  # Region was empty or fully overridden

                # Check if this region encompasses all atoms in the structure
                num_tagged = np.sum(tags == tag_id)
                is_all = num_tagged == len(atoms)

                group_name = f"region_{tag_id}_group"

                if is_all:
                    # Optimize to use 'all' group
                    group_name = "all"
                else:
                    # Explicit region and group definition
                    region_name = f"spatial_region_{tag_id}"
                    spatial_commands.append(
                        f"region {region_name} block {region.x_min} {region.x_max} {region.y_min} {region.y_max} {region.z_min} {region.z_max} units box"
                    )
                    spatial_commands.append(f"group {group_name} region {region_name}")

                # Apply fix based on action
                if region.action == SpatialAction.ACTION_FREEZE:
                    spatial_commands.append(
                        f"fix freeze_fix_{tag_id} {group_name} setforce 0.0 0.0 0.0"
                    )
                elif region.action == SpatialAction.ACTION_LANGEVIN_THERMOSTAT:
                    from pyacemaker.domain_models.defaults import (
                        DEFAULT_LANGEVIN_DAMPING,
                        DEFAULT_LANGEVIN_SEED,
                        DEFAULT_LANGEVIN_TEMP,
                    )

                    spatial_commands.append(
                        f"fix langevin_fix_{tag_id} {group_name} langevin {DEFAULT_LANGEVIN_TEMP} {DEFAULT_LANGEVIN_TEMP} {DEFAULT_LANGEVIN_DAMPING} {DEFAULT_LANGEVIN_SEED}"
                    )
                elif region.action == SpatialAction.ACTION_ACTIVE_LEARNING_ONLY:
                    # This might just define a group for later use
                    pass

        from pyacemaker.domain_models.defaults import DEFAULT_SUPERCELL_SIZE

        return StructureConfig(
            elements=[data.chemical_symbol], supercell_size=DEFAULT_SUPERCELL_SIZE
        ), spatial_commands

    @classmethod
    def _compile_mace_training_node(
        cls,
        material: str,
    ) -> TrainingConfig:
        from pyacemaker.domain_models.defaults import (
            DEFAULT_MACE_BATCH_SIZE,
            DEFAULT_TRAINING_CUTOFF_RADIUS,
            DEFAULT_TRAINING_MAX_BASIS_SIZE,
            DEFAULT_TRAINING_MAX_ITERATIONS,
        )

        # Intelligent defaults
        return TrainingConfig(
            potential_type="mace",
            cutoff_radius=DEFAULT_TRAINING_CUTOFF_RADIUS,
            max_basis_size=DEFAULT_TRAINING_MAX_BASIS_SIZE,
            pacemaker=PacemakerConfig(),
            elements=[material],
            delta_learning=False,
            active_set_optimization=False,
            active_set_size=None,
            max_iterations=DEFAULT_TRAINING_MAX_ITERATIONS,
            batch_size=DEFAULT_MACE_BATCH_SIZE,
        )

    @classmethod
    def _compile_active_learning_node(
        cls,
        slider: int,
        material: str,
        spatial_commands: list[str] | None = None,
    ) -> tuple[MDConfig, DFTConfig, WorkflowConfig]:
        # Intelligent defaults mapping based on material and slider
        from pyacemaker.domain_models.defaults import (
            DEFAULT_FIX_HALT,
            DEFAULT_MASS_THRESHOLD_HEAVY,
            DEFAULT_MASS_THRESHOLD_LIGHT,
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_MD_PRESSURE,
            DEFAULT_MD_TEMPERATURE,
            DEFAULT_MD_UNITS,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
            DEFAULT_SOFT_START_LANGEVIN_DAMP,
            DEFAULT_SOFT_START_STEPS,
            DEFAULT_TIMESTEP_BASE,
            DEFAULT_TIMESTEP_HEAVY,
            DEFAULT_TIMESTEP_LIGHT,
        )

        # 1. Determine base mass and timestep
        if material not in chemical_symbols:
            msg = "Invalid chemical symbol"
            raise CompilerError(msg)

        if not isinstance(slider, int) or not (DEFAULT_SLIDER_MIN <= slider <= DEFAULT_SLIDER_MAX):
            msg = f"Slider must be an integer between {DEFAULT_SLIDER_MIN} and {DEFAULT_SLIDER_MAX}"
            raise CompilerError(msg)

        z = chemical_symbols.index(material)
        mass = atomic_masses[z]

        # Lighter elements -> smaller timestep.
        timestep = DEFAULT_TIMESTEP_BASE
        if mass > DEFAULT_MASS_THRESHOLD_HEAVY:
            timestep = DEFAULT_TIMESTEP_HEAVY
        if mass < DEFAULT_MASS_THRESHOLD_LIGHT:
            timestep = DEFAULT_TIMESTEP_LIGHT

        from pyacemaker.domain_models.defaults import (
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
        )

        # Scale thresholds based on accuracy slider
        # slider MAX -> stricter
        slider_range = max(1, DEFAULT_SLIDER_MAX - DEFAULT_SLIDER_MIN)
        conv_thr = 1e-5 + (1e-4 - 1e-5) * (DEFAULT_SLIDER_MAX - slider) / float(slider_range)
        conv_thr = max(conv_thr, 1e-6)

        # Higher slider -> more accurate -> lower threshold to call DFT
        call_dft_thr = 0.01 + 0.1 * (DEFAULT_SLIDER_MAX - slider) / float(slider_range)

        if spatial_commands is not None:
            import re

            from pyacemaker.domain_models.constants import LAMMPS_SAFE_CMD_PATTERN

            pattern = re.compile(LAMMPS_SAFE_CMD_PATTERN)
            for cmd in spatial_commands:
                if not pattern.match(cmd):
                    msg = f"Invalid spatial command generated: {cmd}"
                    raise CompilerError(msg)

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
            fix_halt=DEFAULT_FIX_HALT,
            uncertainty_threshold=call_dft_thr,
            check_interval=10,
            ramping=None,
            mc=None,
            soft_start_steps=DEFAULT_SOFT_START_STEPS,
            soft_start_langevin_damp=DEFAULT_SOFT_START_LANGEVIN_DAMP,
            custom_initialization_commands=spatial_commands,
        )

        from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_CODE,
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_FUNCTIONAL,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
            DEFAULT_DFT_SMEARING_TYPE,
            DEFAULT_DFT_SMEARING_WIDTH,
            DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            DEFAULT_PSEUDOPOTENTIAL_MAPPING,
        )

        safe_pseudo = DEFAULT_PSEUDOPOTENTIAL_MAPPING.get(material)
        if not safe_pseudo:
            msg = f"No verified pseudopotential mapping exists for material: {material}"
            raise CompilerError(msg)

        from pyacemaker.domain_models.defaults import (
            DEFAULT_ENCUT_BASE,
            DEFAULT_ENCUT_FACTOR,
            DEFAULT_KPOINTS_DENSITY_BASE,
            DEFAULT_KPOINTS_DENSITY_FACTOR,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
        )

        slider_range = max(1, DEFAULT_SLIDER_MAX - DEFAULT_SLIDER_MIN)

        dft_config = DFTConfig(
            code=DEFAULT_DFT_CODE,
            functional=DEFAULT_DFT_FUNCTIONAL,
            kpoints_density=DEFAULT_KPOINTS_DENSITY_BASE
            + DEFAULT_KPOINTS_DENSITY_FACTOR * (DEFAULT_SLIDER_MAX - slider) / float(slider_range),
            encut=DEFAULT_ENCUT_BASE + (slider * DEFAULT_ENCUT_FACTOR),
            pseudopotentials={material: safe_pseudo},
            embedding_buffer=None,
            mixing_beta=DEFAULT_DFT_MIXING_BETA,
            smearing_type=DEFAULT_DFT_SMEARING_TYPE,
            smearing_width=DEFAULT_DFT_SMEARING_WIDTH,
            diagonalization=DEFAULT_DFT_DIAGONALIZATION,
            mixing_beta_factor=DEFAULT_DFT_MIXING_BETA_FACTOR,
            smearing_width_factor=DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
        )

        workflow_config = WorkflowConfig(
            max_iterations=DEFAULT_MAX_ITERATIONS,
            loop_strategy=LoopStrategyConfig(
                thresholds=ActiveLearningThresholds(
                    threshold_call_dft=call_dft_thr, threshold_add_train=call_dft_thr / 2.0
                )
            ),
        )

        return md_config, dft_config, workflow_config
