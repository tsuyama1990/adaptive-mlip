import re

with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

# Fix import
content = content.replace("from ase.io import read, write", "import os\nimport random\nfrom collections import deque\nfrom ase.io import read, write, iread")

# Fix PacemakerTrainer path resolving and executable
pacemaker_train = """
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        \"\"\"
        Trains a potential using the provided training data file.

        This method wraps the external 'pace_train' command.
        It generates 'input.yaml' configuration for Pacemaker and executes the training.

        Args:
            training_data_path: Path to the file containing labelled structures.
                                Supported formats: .pckl, .xyz, .extxyz, .gzip.
            initial_potential: Optional path to an existing potential to fine-tune from.

        Returns:
            Path: The path to the generated potential file (e.g., potential.yace).

        Raises:
            TrainerError: If the training data file does not exist or format is invalid.
        \"\"\"
        # Get executable from env var, defaulting to pace_train
        executable = os.environ.get("PACE_TRAIN_EXECUTABLE", "pace_train")
        if not shutil.which(executable):
            msg = f"Executable '{executable}' not found in PATH."
            raise TrainerError(msg)

        data_path = Path(training_data_path).resolve(strict=True)
        self._validate_training_data(data_path)

        # Determine output directory (same as data file)
        output_dir = data_path.parent
        input_yaml_path = output_dir / "input.yaml"
        potential_path = output_dir / self.config.output_filename

        # Generate configuration
        pacemaker_config = self.config_generator.generate(str(data_path), str(potential_path))
        dump_yaml(pacemaker_config, input_yaml_path)

        # Run pace_train safely
        cmd = [executable, str(input_yaml_path)]

        if initial_potential:
            initial_path = Path(initial_potential).resolve(strict=True)
            cmd.extend(["--initial_potential", str(initial_path)])

        try:
            # Use shell=False implicitly (default for list commands)
            run_command(cmd)
"""
content = re.sub(r'    def train\([\s\S]*?run_command\(cmd\)', pacemaker_train, content)

# Fix IncrementalTrainer logic
inc_trainer = """
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:
        \"\"\"
        Trains a potential incrementally.

        1. Reads new structures from training_data_path.
        2. Appends to master history file (training_history.extxyz) without loading history into memory.
        3. Samples up to replay_buffer_size from history using a bounded deque to prevent OOM.
        4. Writes sampled structures to a temporary training set.
        5. Calls base_trainer.train with the temporary set and initial_potential.
        \"\"\"
        data_path = Path(training_data_path).resolve(strict=True)
        output_dir = data_path.parent
        history_path = output_dir / "training_history.extxyz"
        temp_train_path = output_dir / "training_set_temp.extxyz"

        # Read new structures (assume new_structures is small enough for memory, as it's just candidates)
        new_structures = list(read(str(data_path), index=":"))

        # Append new structures to history file efficiently
        write(str(history_path), new_structures, format="extxyz", append=True)

        # Replay buffer sampling via reservoir sampling or bounded deque
        # Using a simple fixed-size bounded deque over streaming iread to handle memory safety
        buffer = deque(maxlen=self.replay_buffer_size)
        try:
            for frame in iread(str(history_path), format="extxyz"):
                buffer.append(frame)
        except Exception:
            pass

        sampled_structures = list(buffer)

        # Write out temp training set
        write(str(temp_train_path), sampled_structures, format="extxyz")

        # Train using base trainer
        if initial_potential:
            initial_potential = Path(initial_potential).resolve(strict=True)

        return self.base_trainer.train(temp_train_path, initial_potential=initial_potential)
"""
content = re.sub(r'    def train\(\s+self, training_data_path: str \| Path, initial_potential: str \| Path \| None = None\s+\) -> Any:[\s\S]*?return self.base_trainer.train\(temp_train_path, initial_potential=initial_potential\)', inc_trainer, content)

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
