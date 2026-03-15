with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

# I see it generated: 'region reg_2 block 0.0 5.0 0.0 5.0 3.0 5.0 units box'
# But my assert was `any("region reg_2 block 0.0 5.0" in cmd for cmd in cmds)` which IS true for `region reg_2 block 0.0 5.0 0.0 5.0 3.0 5.0 units box`
# Wait, why did `assert any("region reg_2 block 0.0 5.0" in cmd for cmd in cmds)` fail?
# Let's see the printed cmds!
