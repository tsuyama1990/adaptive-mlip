with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

content = content.replace('    # Check if cmds are present\n    assert len(cmds) > 0\n    assert any("group" in cmd for cmd in cmds)', '    # Check if cmds are present\n    assert len(cmds) > 0\n    assert any("group" in cmd for cmd in cmds)\n    assert any("region" in cmd for cmd in cmds)\n    assert any("fix" in cmd for cmd in cmds)')

with open("tests/integration/test_compiler.py", "w") as f:
    f.write(content)
