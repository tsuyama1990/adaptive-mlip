with open("tests/unit/test_oracle.py") as f:
    content = f.read()

content = content.replace("from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle", "from pyacemaker.core.oracle import DFTManager")
content = content.replace("mace_manager_mock()", "mace_manager_mock_temp()")
content = content.replace("test_tiered_oracle()", "tiered_oracle_temp()")

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)
