with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# The error matches are not bubbling up as OrchestratorError in the new distillation code.
# Let's fix Orchestrator._check_initial_potential to catch the exception and raise OrchestratorError
