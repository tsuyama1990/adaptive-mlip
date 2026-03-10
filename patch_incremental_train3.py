with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

# Wait, the `train` method is checking suffix. A named pipe might not have a suffix unless we name it with one.
# We named it "stream.extxyz" so that is handled.
# One issue is `data_path.stat().st_size == 0` check in `_validate_training_data`.
# A named pipe will have a size of 0.
content = content.replace(
"""        # Check for empty file
        if data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise TrainerError(msg)""",
"""        # Check for empty file
        # Named pipes (FIFOs) have size 0, so skip the check for them
        if data_path.is_file() and data_path.stat().st_size == 0:
            msg = f"Training data file is empty: {data_path}"
            raise TrainerError(msg)""")

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
