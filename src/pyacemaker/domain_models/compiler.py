from typing import Any, cast

import ase.build
import networkx as nx
from ase.data import chemical_symbols

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.dft import DFTConfig
from pyacemaker.domain_models.heuristics import HeuristicConfigDict
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

        slider = intent.accuracy_speed_slider
        material = intent.target_material
        overrides = intent.advanced_settings or {}

        # 3. Heuristics calculation
        from pyacemaker.domain_models.heuristics import get_heuristics_for_slider

        heuristics_cfg = get_heuristics_for_slider(slider, [material])

        # Build components
        structure_config = None
        training_config = None
        md_config = None
        dft_config = None
        workflow_config = None

        spatial_commands: list[str] | None = None

        for node in sorted_nodes:
            match node.type:
                case NodeType.INITIAL_STRUCTURE:
                    structure_config, spatial_commands = cls._compile_initial_structure_node(node)
                case NodeType.MACE_TRAINING:
                    training_config = cls._compile_mace_training_node(
                        material, heuristics_cfg, overrides
                    )
                case NodeType.ACTIVE_LEARNING_LOOP:
                    md_config, dft_config, workflow_config = cls._compile_active_learning_node(
                        slider, material, spatial_commands, heuristics_cfg, overrides
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
        from pyacemaker.domain_models.heuristics import get_heuristics_for_slider

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
        heuristics: "HeuristicConfigDict",
        overrides: dict[str, Any],
    ) -> TrainingConfig:
        from pyacemaker.domain_models.defaults import (
            DEFAULT_MACE_BATCH_SIZE,
            DEFAULT_TRAINING_CUTOFF_RADIUS,
            DEFAULT_TRAINING_MAX_BASIS_SIZE,
            DEFAULT_TRAINING_MAX_ITERATIONS,
        )

        # Base defaults
        training_kwargs: dict[str, Any] = {
            "potential_type": "mace",
            "cutoff_radius": DEFAULT_TRAINING_CUTOFF_RADIUS,
            "max_basis_size": DEFAULT_TRAINING_MAX_BASIS_SIZE,
            "elements": [material],
            "delta_learning": False,
            "active_set_optimization": False,
            "active_set_size": None,
            "max_iterations": DEFAULT_TRAINING_MAX_ITERATIONS,
            "batch_size": DEFAULT_MACE_BATCH_SIZE,
        }

        pacemaker_kwargs: dict[str, Any] = {}

        # Heuristics
        if "training" in heuristics and "pacemaker" in heuristics["training"]:
            pacemaker_kwargs.update(heuristics["training"]["pacemaker"])

        # Overrides
        # We allow overrides to directly provide 'learning_rate' or 'max_iterations' etc
        for k, v in overrides.items():
            if k in training_kwargs:
                training_kwargs[k] = v
            # If user provides explicit pacemaker args
            if hasattr(PacemakerConfig, k) or k == "learning_rate":
                pacemaker_kwargs[k] = v

        if pacemaker_kwargs:
            training_kwargs["pacemaker"] = PacemakerConfig(**pacemaker_kwargs)
        else:
            training_kwargs["pacemaker"] = PacemakerConfig()

        return TrainingConfig(**training_kwargs)

    @classmethod
    def _get_base_md_kwargs(cls, spatial_commands: list[str] | None) -> dict[str, Any]:
        from pyacemaker.domain_models.defaults import (
            DEFAULT_FIX_HALT,
            DEFAULT_MD_PRESSURE,
            DEFAULT_MD_TEMPERATURE,
            DEFAULT_MD_UNITS,
            DEFAULT_SOFT_START_LANGEVIN_DAMP,
            DEFAULT_SOFT_START_STEPS,
        )

        return {
            "temperature": DEFAULT_MD_TEMPERATURE,
            "pressure": DEFAULT_MD_PRESSURE,
            "n_steps": 1000,
            "units": DEFAULT_MD_UNITS,
            "atom_style": AtomStyle.ATOMIC,
            "thermo_freq": 100,
            "dump_freq": 100,
            "minimize": False,
            "neighbor_skin": 2.0,
            "tdamp_factor": 100.0,
            "pdamp_factor": 1000.0,
            "hybrid_potential": False,
            "fix_halt": DEFAULT_FIX_HALT,
            "ramping": None,
            "mc": None,
            "soft_start_steps": DEFAULT_SOFT_START_STEPS,
            "soft_start_langevin_damp": DEFAULT_SOFT_START_LANGEVIN_DAMP,
            "custom_initialization_commands": spatial_commands,
        }

    @classmethod
    def _get_base_dft_kwargs(cls, slider: int, material: str) -> dict[str, Any]:
        from pyacemaker.domain_models.defaults import (
            DEFAULT_DFT_CODE,
            DEFAULT_DFT_DIAGONALIZATION,
            DEFAULT_DFT_FUNCTIONAL,
            DEFAULT_DFT_MIXING_BETA,
            DEFAULT_DFT_MIXING_BETA_FACTOR,
            DEFAULT_DFT_SMEARING_TYPE,
            DEFAULT_DFT_SMEARING_WIDTH,
            DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            DEFAULT_ENCUT_BASE,
            DEFAULT_ENCUT_FACTOR,
            DEFAULT_KPOINTS_DENSITY_BASE,
            DEFAULT_KPOINTS_DENSITY_FACTOR,
            DEFAULT_PSEUDOPOTENTIAL_MAPPING,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
        )

        safe_pseudo = DEFAULT_PSEUDOPOTENTIAL_MAPPING.get(material)
        if not safe_pseudo:
            msg = f"No verified pseudopotential mapping exists for material: {material}"
            raise CompilerError(msg)

        slider_range = max(1, DEFAULT_SLIDER_MAX - DEFAULT_SLIDER_MIN)
        return {
            "code": DEFAULT_DFT_CODE,
            "functional": DEFAULT_DFT_FUNCTIONAL,
            "kpoints_density": DEFAULT_KPOINTS_DENSITY_BASE
            + DEFAULT_KPOINTS_DENSITY_FACTOR * (DEFAULT_SLIDER_MAX - slider) / float(slider_range),
            "pseudopotentials": {material: safe_pseudo},
            "embedding_buffer": None,
            "mixing_beta": DEFAULT_DFT_MIXING_BETA,
            "diagonalization": DEFAULT_DFT_DIAGONALIZATION,
            "mixing_beta_factor": DEFAULT_DFT_MIXING_BETA_FACTOR,
            "smearing_width_factor": DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            "encut": DEFAULT_ENCUT_BASE + (slider * DEFAULT_ENCUT_FACTOR),
            "smearing_type": DEFAULT_DFT_SMEARING_TYPE,
            "smearing_width": DEFAULT_DFT_SMEARING_WIDTH,
        }

    @classmethod
    def _apply_heuristics_and_overrides(  # noqa: C901
        cls,
        md_kwargs: dict[str, Any],
        dft_kwargs: dict[str, Any],
        workflow_kwargs: dict[str, Any],
        heuristics: "HeuristicConfigDict",
        overrides: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        loop_strategy_kwargs: dict[str, Any] = {}
        thresholds_kwargs: dict[str, Any] = {}

        if "md" in heuristics:
            md_kwargs.update(heuristics["md"])
        if "dft" in heuristics:
            dft_kwargs.update(heuristics["dft"])
        if "workflow" in heuristics:
            wf_h = heuristics["workflow"]
            if "loop_strategy" in wf_h and "thresholds" in wf_h["loop_strategy"]:
                thresholds_kwargs.update(wf_h["loop_strategy"]["thresholds"])

        mapped_overrides = overrides.copy()
        if "ecutwfc" in mapped_overrides:
            mapped_overrides["encut"] = mapped_overrides.pop("ecutwfc")

        for k, v in mapped_overrides.items():
            if k in md_kwargs:
                md_kwargs[k] = v
            elif k in dft_kwargs:
                dft_kwargs[k] = v
            elif k in workflow_kwargs:
                workflow_kwargs[k] = v
            elif k in ["threshold_call_dft", "threshold_add_train"]:
                thresholds_kwargs[k] = v

        return loop_strategy_kwargs, thresholds_kwargs

    @classmethod
    def _compile_active_learning_node(  # noqa: C901, PLR0912, PLR0915
        cls,
        slider: int,
        material: str,
        spatial_commands: list[str] | None,
        heuristics: "HeuristicConfigDict",
        overrides: dict[str, Any],
    ) -> tuple[MDConfig, DFTConfig, WorkflowConfig]:
        from pyacemaker.domain_models.defaults import (
            DEFAULT_FIX_HALT,
            DEFAULT_MD_PRESSURE,
            DEFAULT_MD_TEMPERATURE,
            DEFAULT_MD_UNITS,
            DEFAULT_SLIDER_MAX,
            DEFAULT_SLIDER_MIN,
            DEFAULT_SOFT_START_LANGEVIN_DAMP,
            DEFAULT_SOFT_START_STEPS,
        )

        if material not in chemical_symbols:
            msg = "Invalid chemical symbol"
            raise CompilerError(msg)

        if not isinstance(slider, int) or not (DEFAULT_SLIDER_MIN <= slider <= DEFAULT_SLIDER_MAX):
            msg = f"Slider must be an integer between {DEFAULT_SLIDER_MIN} and {DEFAULT_SLIDER_MAX}"
            raise CompilerError(msg)

        if spatial_commands is not None:
            import re

            from pyacemaker.domain_models.constants import LAMMPS_SAFE_CMD_PATTERN

            pattern = re.compile(LAMMPS_SAFE_CMD_PATTERN)
            for cmd in spatial_commands:
                if not pattern.match(cmd):
                    msg = f"Invalid spatial command generated: {cmd}"
                    raise CompilerError(msg)

        # 1. Base MD
        md_kwargs: dict[str, Any] = {
            "temperature": DEFAULT_MD_TEMPERATURE,
            "pressure": DEFAULT_MD_PRESSURE,
            "n_steps": 1000,
            "units": DEFAULT_MD_UNITS,
            "atom_style": AtomStyle.ATOMIC,
            "thermo_freq": 100,
            "dump_freq": 100,
            "minimize": False,
            "neighbor_skin": 2.0,
            "tdamp_factor": 100.0,
            "pdamp_factor": 1000.0,
            "hybrid_potential": False,
            "fix_halt": DEFAULT_FIX_HALT,
            "ramping": None,
            "mc": None,
            "soft_start_steps": DEFAULT_SOFT_START_STEPS,
            "soft_start_langevin_damp": DEFAULT_SOFT_START_LANGEVIN_DAMP,
            "custom_initialization_commands": spatial_commands,
        }

        # Base DFT
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
        )

        slider_range = max(1, DEFAULT_SLIDER_MAX - DEFAULT_SLIDER_MIN)

        dft_kwargs: dict[str, Any] = {
            "code": DEFAULT_DFT_CODE,
            "functional": DEFAULT_DFT_FUNCTIONAL,
            "kpoints_density": DEFAULT_KPOINTS_DENSITY_BASE
            + DEFAULT_KPOINTS_DENSITY_FACTOR * (DEFAULT_SLIDER_MAX - slider) / float(slider_range),
            "pseudopotentials": {material: safe_pseudo},
            "embedding_buffer": None,
            "mixing_beta": DEFAULT_DFT_MIXING_BETA,
            "diagonalization": DEFAULT_DFT_DIAGONALIZATION,
            "mixing_beta_factor": DEFAULT_DFT_MIXING_BETA_FACTOR,
            "smearing_width_factor": DEFAULT_DFT_SMEARING_WIDTH_FACTOR,
            # Will be filled by heuristics if present:
            "encut": DEFAULT_ENCUT_BASE + (slider * DEFAULT_ENCUT_FACTOR),
            "smearing_type": DEFAULT_DFT_SMEARING_TYPE,
            "smearing_width": DEFAULT_DFT_SMEARING_WIDTH,
        }

        # Base Workflow
        from pyacemaker.domain_models.defaults import DEFAULT_MAX_ITERATIONS

        workflow_kwargs: dict[str, Any] = {
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        }
        loop_strategy_kwargs: dict[str, Any] = {}
        thresholds_kwargs: dict[str, Any] = {}

        # 2. Heuristics Merging
        if "md" in heuristics:
            md_kwargs.update(heuristics["md"])
        if "dft" in heuristics:
            dft_kwargs.update(heuristics["dft"])
        if "workflow" in heuristics:
            wf_h = heuristics["workflow"]
            if "loop_strategy" in wf_h and "thresholds" in wf_h["loop_strategy"]:
                thresholds_kwargs.update(wf_h["loop_strategy"]["thresholds"])

        # 3. Overrides Merging
        # Alias mapping handling: ecutwfc -> encut
        mapped_overrides = overrides.copy()
        if "ecutwfc" in mapped_overrides:
            mapped_overrides["encut"] = mapped_overrides.pop("ecutwfc")

        for k, v in mapped_overrides.items():
            if k in md_kwargs:
                md_kwargs[k] = v
            elif k in dft_kwargs:
                dft_kwargs[k] = v
            elif k in workflow_kwargs:
                workflow_kwargs[k] = v
            elif k in ["threshold_call_dft", "threshold_add_train"]:
                thresholds_kwargs[k] = v
            # other keys might be ignored or handled if we want to expand

        # Assemble final nested
        if thresholds_kwargs:
            loop_strategy_kwargs["thresholds"] = ActiveLearningThresholds(**thresholds_kwargs)
        if loop_strategy_kwargs:
            workflow_kwargs["loop_strategy"] = LoopStrategyConfig(**loop_strategy_kwargs)

        md_config = MDConfig(**md_kwargs)
        dft_config = DFTConfig(**dft_kwargs)
        workflow_config = WorkflowConfig(**workflow_kwargs)

        return md_config, dft_config, workflow_config
