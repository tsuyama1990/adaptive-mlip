import re

with open("tests/unit/test_oracle.py") as f:
    content = f.read()

# Fix empty iterator logic by reverting to checking if an empty iterator runs correctly
# Since `with pytest.warns(None)` caused TypeError in pytest, we just test the result
content = re.sub(
    r'        from collections import deque\n        with pytest.warns\(None\): # the empty iterator logic changed to just return empty list',
    '        result = list(manager.compute(empty_iter))\n        assert len(result) == 0',
    content
)

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)
