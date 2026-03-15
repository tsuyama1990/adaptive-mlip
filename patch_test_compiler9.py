with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

content = content.replace('    assert len(cmds) > 0', '    # Debugging\n    print("CMDS:", cmds)\n    assert len(cmds) > 0')

with open("tests/integration/test_compiler.py", "w") as f:
    f.write(content)
