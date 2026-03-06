
with open("tests/unit/test_oracle.py") as f:
    content = f.read()

# Replace with the logic from the master test which explicitly tests the warning mechanism we originally bypassed
# but it was broken by our refactoring of DFTManager.
# Our new DFTManager implementation returns a generator that doesn't yield anything and doesn't explicitly raise UserWarning.
# We will explicitly add a warning if list is empty to pass the test.

content = content.replace("""        result = list(manager.compute(empty_iter))
        assert len(result) == 0""", """        with pytest.warns(UserWarning, match="Oracle received empty iterator"):
            list(manager.compute(empty_iter))""")

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)
