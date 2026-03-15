with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

content = content.replace('    assert any("region reg_2 block 0.0 5.0" in cmd for cmd in cmds)', '    # Just check it exists\n    assert any("region reg_2 block 0.0 5.0" in cmd for cmd in cmds)')

with open("tests/integration/test_compiler.py", "w") as f:
    f.write(content)
