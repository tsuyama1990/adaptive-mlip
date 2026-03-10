with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

search = """class FinetuneManager:
    \"\"\"
    Manager to briefly train the final readout layers of the MACE foundation model.
    \"\"\"

    def finetune(self, dataset_path: str | Path) -> str:
        \"\"\"
        Mock finetuning logic for the awakened MACE model.
        Returns the path to the awakened model.
        \"\"\"
        return "awakened_mace_model.model\""""

replace = """class FinetuneManager:
    \"\"\"
    Manager to briefly train the final readout layers of the MACE foundation model.
    \"\"\"

    def finetune(self, dataset_path: str | Path) -> str:
        \"\"\"
        Finetunes the awakened MACE model using the provided dataset.
        Returns the path to the awakened model.
        \"\"\"
        import subprocess
        from pyacemaker.utils.process import run_command
        import logging

        logger = logging.getLogger(__name__)
        output_model = "awakened_mace_model.model"

        # Read foundation model from env or use default
        import os
        foundation_model = os.environ.get("MACE_FOUNDATION_MODEL", "mace-mp-0-medium")
        mace_train_cmd = "mace_run_train"

        cmd = [
            mace_train_cmd,
            "--name",
            "awakened_mace_model",
            "--train_file",
            str(dataset_path),
            "--foundation_model",
            foundation_model,
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"MACE Finetuning failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            msg = f"MACE Finetuning failed unexpectedly: {e}"
            raise TrainerError(msg) from e

        return output_model"""

content = content.replace(search, replace)

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
