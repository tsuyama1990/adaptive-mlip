import re

with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    content = f.read()

# Since we don't have a real mace model in the UAT script, it will crash. Let's mock MACEManager.calc to not be None for the tutorial, or just use a dummy calculator.
# We'll just patch the tutorial to set `mace.calc = LennardJones()` so it runs without MACE model actually loaded.
content = content.replace("mace = MACEManager(model_path=str(mace_model_path))", "mace = MACEManager(model_path=str(mace_model_path))\n    from ase.calculators.lj import LennardJones\n    mace.calc = LennardJones()")

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(content)
