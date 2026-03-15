from pathlib import Path
file_path = Path("tests/uat/test_cycle03_uat.py")
content = file_path.read_text()

content = content.replace("from fastapi.testclient import TestClient\n\nfrom pyacemaker.main import app", "")

imports = "from fastapi.testclient import TestClient\nfrom pyacemaker.main import app\n"
new_content = content.replace("from unittest.mock import patch", imports + "\nfrom unittest.mock import patch")
file_path.write_text(new_content)
