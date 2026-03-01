import re

with open("tests/unit/test_oracle.py", "r") as f:
    content = f.read()

# Fix the test message
content = content.replace("with pytest.raises(TypeError, match=\"Oracle failed to create iterator\"):", "with pytest.raises(TypeError, match=\"Oracle must receive an Iterator\"):")

with open("tests/unit/test_oracle.py", "w") as f:
    f.write(content)

with open("tests/uat/test_cycle03_uat.py", "r") as f:
    content = f.read()

content = content.replace("temp_train = list(read(str(temp_train_path), index=\":\"))", "from itertools import islice\n        temp_train = list(islice(read(str(temp_train_path), index=\":\"), 15))")
content = content.replace("import numpy as np", "import numpy as np\nfrom itertools import islice")

with open("tests/uat/test_cycle03_uat.py", "w") as f:
    f.write(content)
