import re

with open("tests/unit/test_lammps_driver_security.py", "r") as f:
    content = f.read()

search = """def test_validate_command_unrecognized(driver):"""

replace = """def test_validate_command_unsafe_chars(driver):
    \"\"\"Test commands with unsafe characters fail.\"\"\"
    unsafe_cmds = [
        "shell ls -la",  # shell token is blocked
        "print 'hello' &",  # & forbidden
        "run 100; rm -rf /",  # ; forbidden
        "variable x string `whoami`",  # ` forbidden
        "print 'hello' | grep x",  # | forbidden
    ]
    for cmd in unsafe_cmds:
        with pytest.raises(
            ValueError, match="forbidden characters|forbidden command|unrecognized command"
        ):
            driver._validate_command(cmd)

def test_validate_command_unrecognized(driver):"""

content = content.replace(search, replace)

with open("tests/unit/test_lammps_driver_security.py", "w") as f:
    f.write(content)
