import re
from pathlib import Path

# Fix the warning test again by making sure our mock empty iterator test actually succeeds and warns
# If the warning fails, it means we don't emit the warning when batch is totally empty.
# We DID add the warning to our implementation! Wait, `batch = list(islice(structures, batch_size))`
# If it's totally empty from the start, `not batch` is True, `first_batch` is True, it warns.
# Let's ensure the test correctly patches/matches the text.
test_p = Path("tests/unit/test_oracle.py")
test_c = test_p.read_text()
test_c = test_c.replace(
    '        import warnings\n        with warnings.catch_warnings():\n            warnings.simplefilter("ignore")',
    '        with pytest.warns(UserWarning, match="Oracle received empty iterator"): \n            deque(manager.compute(empty_iter), maxlen=0)'
)
test_p.write_text(test_c)

# We should make sure the warning is imported globally correctly in core/oracle.py just in case
p_oracle = Path("src/pyacemaker/core/oracle.py")
oracle_c = p_oracle.read_text()
if "import warnings" not in oracle_c:
    oracle_c = "import warnings\n" + oracle_c
    p_oracle.write_text(oracle_c)
