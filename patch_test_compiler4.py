with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

content = content.replace('print(cmds)\\n', 'print("\\n\\nCMDS GENERATED:\\n", cmds)\\n')

with open("tests/integration/test_compiler.py", "w") as f:
    f.write(content)
