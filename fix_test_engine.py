with open("tests/unit/test_engine.py", "r") as f:
    text = f.read()

# Add patch for pyacemaker.core.io_manager.validate_path_safe inside test_run_large_structure_warning
replacement = """    with (
        patch("pyacemaker.core.io_manager.write_lammps_streaming") as mock_stream,
        patch("pyacemaker.core.io_manager.get_species_order", return_value=["H"]),
        patch("pyacemaker.utils.path.Path.lstat") as mock_lstat,
        patch("pyacemaker.utils.path.Path.stat") as mock_stat,
        patch("pyacemaker.core.io_manager.validate_path_safe", side_effect=lambda x: x),
    ):"""

text = text.replace("""    with (
        patch("pyacemaker.core.io_manager.write_lammps_streaming") as mock_stream,
        patch("pyacemaker.core.io_manager.get_species_order", return_value=["H"]),
        patch("pyacemaker.utils.path.Path.lstat") as mock_lstat,
        patch("pyacemaker.utils.path.Path.stat") as mock_stat,
    ):""", replacement)

with open("tests/unit/test_engine.py", "w") as f:
    f.write(text)
