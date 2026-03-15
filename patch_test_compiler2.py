with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

content = content.replace('    assert any("region reg_1 block 0.0 5.0" in cmd for cmd in cmds)', '    # The x_max is 5.0, LAMMPS uses xlo xhi ylo yhi zlo zhi\\n    assert any("region reg_1 block 0.0 5.0" in cmd for cmd in cmds)')
# Wait, let's print `cmds` in the test to see what it is actually generating.

with open("tests/integration/test_compiler.py", "w") as f:
    f.write(content.replace("assert any", "print(cmds)\\n    assert any", 1))
