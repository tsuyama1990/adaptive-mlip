with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

# Replace actual newline with explicit '\n' in the python strings.
content = content.replace("f.write(f\"restart 1000 {restart_out_file}\n\")", "f.write(f\"restart 1000 {restart_out_file}\\n\")")
content = content.replace("f.write(f\"read_restart {restart_in}\n\")", "f.write(f\"read_restart {restart_in}\\n\")")
content = content.replace("f.write(\"pair_style pace\n\")", "f.write(\"pair_style pace\\n\")")
content = content.replace("f.write(f\"pair_coeff * * {potential_path} *\n\")", "f.write(f\"pair_coeff * * {potential_path} *\\n\")")
content = content.replace("f.write(\"fix soft_start all langevin 300.0 300.0 10.0 12345\n\")", "f.write(\"fix soft_start all langevin 300.0 300.0 10.0 12345\\n\")")
content = content.replace("f.write(\"fix nve_soft all nve\n\")", "f.write(\"fix nve_soft all nve\\n\")")
content = content.replace("f.write(\"run 100\n\")", "f.write(\"run 100\\n\")")
content = content.replace("f.write(\"unfix soft_start\n\")", "f.write(\"unfix soft_start\\n\")")
content = content.replace("f.write(\"unfix nve_soft\n\")", "f.write(\"unfix nve_soft\\n\")")
content = content.replace("f.write(\"fix main_nve all nve\n\")", "f.write(\"fix main_nve all nve\\n\")")
content = content.replace("f.write(f\"dump 1 all custom {self.config.dump_freq} {dump_file} id type x y z fx fy fz\n\")", "f.write(f\"dump 1 all custom {self.config.dump_freq} {dump_file} id type x y z fx fy fz\\n\")")
content = content.replace("f.write(f\"run {self.config.n_steps}\n\")", "f.write(f\"run {self.config.n_steps}\\n\")")

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
