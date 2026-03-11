with open("tests/e2e/test_orchestrator.py") as f:
    content = f.read()

# The test_integration_workflow_complete has an assertion error.
# The issue might be `assert final_potential.exists()`, but we mock `mock_trainer.train.return_value = str(tmp_path / "fake_potential.yace")`
# However, the code creates it when we mocked `write` maybe?
# The orchestrator copies the potential: `shutil.copy(self.state_manager.current_potential, potential_target)`
# But the fake_potential file doesn't actually exist on disk because `mock_trainer.train` just returns a string, it doesn't touch the file.
# So `shutil.copy` fails if the fake file is not touched.
# Let's fix test_integration_workflow_complete so `train` touches the file before returning.

content = content.replace(
    'mock_trainer.train.return_value = str(tmp_path / "fake_potential.yace")',
    'def mock_train(data, init): p=tmp_path / "fake_potential.yace"; p.touch(); return str(p)\n        mock_trainer.train.side_effect = mock_train',
)

with open("tests/e2e/test_orchestrator.py", "w") as f:
    f.write(content)
