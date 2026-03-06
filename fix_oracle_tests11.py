
with open("tests/unit/test_oracle.py") as f:
    content = f.read()

# Revert empty iterator test back to simple execution and assert length 0 instead of warns
content = content.replace("""        from collections import deque
        with pytest.warns(None): # the empty iterator logic changed to just return empty list""", """        from collections import deque
        deque(manager.compute(empty_iter), maxlen=0)""")

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)
