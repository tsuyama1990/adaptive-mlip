import re

with open('src/pyacemaker/core/generator.py', 'r') as f:
    content = f.read()

# Make sure generate_local propagates to atoms object directly
content = content.replace(
    'atoms_iter = policy.generate(base_structure, self.config, n_structures=n_candidates, **kwargs)',
    'atoms_iter = policy.generate(base_structure, self.config, n_structures=n_candidates, **kwargs)'
)

# This isn't failing right now. The remaining test failures are likely due to the pydantic model config `extra="forbid"`.
