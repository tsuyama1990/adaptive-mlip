import re

with open("tests/unit/test_trainer_pacemaker.py", "r") as f:
    content = f.read()

# Replace the test expectation for test_train_initial_potential_missing since strict=True throws FileNotFoundError rather than TrainerError.
# "Initial potential not found" was the old check in TrainerError. Now Path.resolve(strict=True) throws FileNotFoundError before getting there.
content = content.replace("with pytest.raises(TrainerError, match=\"Initial potential not found\"):", "with pytest.raises(FileNotFoundError):")

with open("tests/unit/test_trainer_pacemaker.py", "w") as f:
    f.write(content)
