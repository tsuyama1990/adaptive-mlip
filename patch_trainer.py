with open("src/pyacemaker/core/trainer.py") as f:
    content = f.read()

# Fix `Training data file is empty` in trainer.py by only checking size if the format is not what it expects or checking if it exists and allowing empty file logic for tests?
# No, we just need to ensure our test writes some data.
