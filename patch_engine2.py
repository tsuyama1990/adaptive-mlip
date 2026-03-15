with open("src/pyacemaker/core/engine.py", "r") as f:
    content = f.read()

target = """                        if np.any(mask):
                            max_gamma = float(np.max(gamma_data[mask]))
                        else:
                            max_gamma = 0.0"""

custom = """                        max_gamma = float(np.max(gamma_data[mask])) if np.any(mask) else 0.0"""

content = content.replace(target, custom)

with open("src/pyacemaker/core/engine.py", "w") as f:
    f.write(content)
