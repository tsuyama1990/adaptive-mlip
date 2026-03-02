with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

content = content.replace("f.write(f\"restart 1000 {restart_out}\n\")", "f.write(f\"restart 1000 {restart_out}\\n\")")

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
