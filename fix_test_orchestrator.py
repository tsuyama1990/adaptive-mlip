with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

content = content.replace("m.setattr(\"pyacemaker.orchestrator.extract_local_region\", mock_fail)", "m.setattr(\"pyacemaker.orchestrator.extract_intelligent_cluster\", mock_fail)")

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
