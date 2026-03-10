with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

content = content.replace("raise TrainerError(msg) from e", "raise TrainerError(msg)")
content = content.replace(
"""        except ValidationError as e:
            msg = f"Generated Pacemaker config failed schema validation: {e}"
            raise TrainerError(msg)""",
"""        except ValidationError as e:
            msg = f"Generated Pacemaker config failed schema validation: {e}"
            raise TrainerError(msg) from e""")
content = content.replace(
"""        except subprocess.CalledProcessError as e:
            # Capture specific subprocess error
            msg = f"Training failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg)
        except Exception as e:
            # Catch other unexpected errors
            msg = f"Training failed unexpectedly: {e}"
            raise TrainerError(msg)""",
"""        except subprocess.CalledProcessError as e:
            # Capture specific subprocess error
            msg = f"Training failed with exit code {e.returncode}: {e}"
            raise TrainerError(msg) from e
        except Exception as e:
            # Catch other unexpected errors
            msg = f"Training failed unexpectedly: {e}"
            raise TrainerError(msg) from e""")


with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
