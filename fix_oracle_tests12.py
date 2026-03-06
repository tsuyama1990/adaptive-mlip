import re

with open("tests/unit/test_oracle.py") as f:
    content = f.read()

# Fix empty iterator logic
# We should completely remove the `with pytest.warns...` from the test
content = re.sub(
    r'        from collections import deque\n        deque\(manager\.compute\(empty_iter\), maxlen=0\)',
    '        result = list(manager.compute(empty_iter))\n        assert len(result) == 0',
    content
)

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)
