import re

files_to_fix = [
    "tests/unit/test_eon_driver.py",
    "tests/unit/test_process.py",
    "tests/unit/mock_process.py",
    "tests/unit/test_report.py",
    "tests/unit/test_telemetry_pubsub.py",
    "tests/unit/test_path_validation.py"
]

for file in files_to_fix:
    try:
        with open(file, "r") as f:
            content = f.read()

        # Fix missing types in test function definitions
        # Find def test_... (var1, var2): and append : Any to untyped vars

        def repl(match):
            args = match.group(2).split(',')
            new_args = []
            for arg in args:
                arg = arg.strip()
                if not arg:
                    continue
                if ':' not in arg:
                    new_args.append(f"{arg}: Any")
                else:
                    new_args.append(arg)
            return f"def {match.group(1)}({', '.join(new_args)}) -> None:"

        content = re.sub(r'def (test_[a-zA-Z0-9_]+)\((.*?)\)(?: -> None)?:', repl, content)

        # Fix class definitions missing imports
        content = content.replace('class MockProcessRunner(ProcessRunner):', 'from pyacemaker.interfaces.process import ProcessRunner\nclass MockProcessRunner(ProcessRunner):')

        with open(file, "w") as f:
            f.write(content)

    except Exception as e:
        print(e)
