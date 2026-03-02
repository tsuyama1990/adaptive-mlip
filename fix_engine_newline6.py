with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

# Just literal replacement
content = content.replace("restart 1000 {restart_out_file}\\\\n", "restart 1000 {restart_out_file}\\n")
content = content.replace("read_restart {restart_in}\\\\n", "read_restart {restart_in}\\n")
content = content.replace("pair_style pace\\\\n", "pair_style pace\\n")
content = content.replace("pair_coeff * * {potential_path} *\\\\n", "pair_coeff * * {potential_path} *\\n")
content = content.replace("fix soft_start all langevin 300.0 300.0 10.0 12345\\\\n", "fix soft_start all langevin 300.0 300.0 10.0 12345\\n")
content = content.replace("fix nve_soft all nve\\\\n", "fix nve_soft all nve\\n")
content = content.replace("run 100\\\\n", "run 100\\n")
content = content.replace("unfix soft_start\\\\n", "unfix soft_start\\n")
content = content.replace("unfix nve_soft\\\\n", "unfix nve_soft\\n")
content = content.replace("fix main_nve all nve\\\\n", "fix main_nve all nve\\n")
content = content.replace("dump 1 all custom {self.config.dump_freq} {dump_file} id type x y z fx fy fz\\\\n", "dump 1 all custom {self.config.dump_freq} {dump_file} id type x y z fx fy fz\\n")
content = content.replace("run {self.config.n_steps}\\\\n", "run {self.config.n_steps}\\n")

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
