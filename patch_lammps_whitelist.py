with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

search = """            "min_style",
            "min_modify",
            "variable",
            "print",
        }"""

replace = """            "min_style",
            "min_modify",
            "variable",
            "print",
            "include",
            "if",
            "jump",
            "label",
            "log",
            "echo",
            "set",
            "group",
            "displace_atoms",
            "write_data",
            "write_restart",
        }"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
