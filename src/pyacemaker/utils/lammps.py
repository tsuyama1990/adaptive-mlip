from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.data import atomic_numbers

from pyacemaker.core.io_manager import LammpsFileManager
from pyacemaker.domain_models.constants import LAMMPS_SCREEN_ARG
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.interfaces.lammps_driver import LammpsDriver


def _generate_base_script(
    data_file: Path,
    potential_path: Path,
    elements: list[str],
    config: MDConfig,
) -> list[str]:
    """Generates base LAMMPS commands."""
    quoted_data = f'"{data_file}"'
    quoted_pot = f'"{potential_path}"'
    species_str = " ".join(elements)

    # Use list building for performance instead of repeated string concatenation
    lines = [
        "clear",
        "units metal",
        f"atom_style {config.atom_style}",
        "boundary p p p",
        f"read_data {quoted_data}",
    ]

    # Potential
    if config.hybrid_potential:
        params = config.hybrid_params
        lines.append(
            f"pair_style hybrid/overlay pace zbl {params.zbl_cut_inner} {params.zbl_cut_outer}"
        )
        lines.append(f"pair_coeff * * pace {quoted_pot} {species_str}")
        n_types = len(elements)
        for i in range(n_types):
            el_i = elements[i]
            z_i = atomic_numbers[el_i]
            for j in range(i, n_types):
                el_j = elements[j]
                z_j = atomic_numbers[el_j]
                lines.append(f"pair_coeff {i + 1} {j + 1} zbl {z_i} {z_j}")
    else:
        lines.append("pair_style pace")
        lines.append(f"pair_coeff * * pace {quoted_pot} {species_str}")

    # Settings
    lines.append(f"neighbor {config.neighbor_skin} bin")
    lines.append("neigh_modify delay 0 every 1 check yes")

    return lines


def _generate_static_script(
    data_file: Path,
    potential_path: Path,
    elements: list[str],
    config: MDConfig,
) -> str:
    """Generates the LAMMPS script for static calculation."""
    lines = _generate_base_script(data_file, potential_path, elements, config)

    # Variables for stress (pressure components in bars)
    lines.extend(
        [
            "variable pxx equal pxx",
            "variable pyy equal pyy",
            "variable pzz equal pzz",
            "variable pxy equal pxy",
            "variable pxz equal pxz",
            "variable pyz equal pyz",
            "run 0",
        ]
    )

    return "\n".join(lines)


def _generate_relax_script(
    data_file: Path,
    potential_path: Path,
    elements: list[str],
    config: MDConfig,
) -> str:
    """Generates the LAMMPS script for structure relaxation."""
    lines = _generate_base_script(data_file, potential_path, elements, config)
    lines.extend(["min_style cg", "minimize 1.0e-6 1.0e-8 1000 10000"])
    return "\n".join(lines)


def run_static_lammps(
    atoms: Atoms,
    potential_path: Path,
    config: MDConfig,
    file_manager: LammpsFileManager | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Runs a static LAMMPS calculation (energy, forces, stress) for the given structure.

    Args:
        atoms: ASE Atoms object.
        potential_path: Path to the potential file.
        config: MDConfig (used for atom style, etc.).
        file_manager: Optional LammpsFileManager.

    Returns:
        Tuple of (energy, forces, stress).
        energy: float (eV)
        forces: np.ndarray (N, 3) (eV/Angstrom)
        stress: np.ndarray (6,) (eV/Angstrom^3) in Voigt notation [xx, yy, zz, yz, xz, xy]
    """
    fm = file_manager or LammpsFileManager(config)

    if not potential_path.exists():
        msg = f"Potential file not found: {potential_path}"
        raise FileNotFoundError(msg)

    potential_path = potential_path.resolve()
    ctx, data_file, _, log_file, elements = fm.prepare_workspace(atoms)
    script = _generate_static_script(data_file, potential_path, elements, config)

    driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
    with ctx:
        try:
            driver.run(script)

            energy = driver.extract_variable("pe")

            # Extract Forces
            f_ptr = driver.lmp.gather_atoms("f", 1, 3)
            natoms = driver.lmp.get_natoms()
            forces = np.ctypeslib.as_array(f_ptr, shape=(natoms, 3)).copy()

            # Reorder forces by ID
            id_ptr = driver.lmp.gather_atoms("id", 0, 1)
            ids = np.ctypeslib.as_array(id_ptr, shape=(natoms,)).copy()
            sorted_forces = np.zeros_like(forces)
            sorted_forces[ids - 1] = forces

            # Extract Stress (Pressure in bars -> Stress in eV/A^3)
            pxx = driver.extract_variable("pxx")
            pyy = driver.extract_variable("pyy")
            pzz = driver.extract_variable("pzz")
            pyz = driver.extract_variable("pyz")
            pxz = driver.extract_variable("pxz")
            pxy = driver.extract_variable("pxy")

            press_bars = np.array([pxx, pyy, pzz, pyz, pxz, pxy])
            stress_ev_a3 = -press_bars * units.bar

        finally:
            driver.lmp.close()

    return energy, sorted_forces, stress_ev_a3


def relax_structure(
    atoms: Atoms,
    potential_path: Path,
    config: MDConfig,
    file_manager: LammpsFileManager | None = None,
) -> Atoms:
    """Relaxes the structure using LAMMPS minimization."""
    fm = file_manager or LammpsFileManager(config)

    if not potential_path.exists():
        msg = f"Potential file not found: {potential_path}"
        raise FileNotFoundError(msg)

    potential_path = potential_path.resolve()
    ctx, data_file, _, log_file, elements = fm.prepare_workspace(atoms)
    script = _generate_relax_script(data_file, potential_path, elements, config)

    driver = LammpsDriver(["-screen", LAMMPS_SCREEN_ARG, "-log", str(log_file)])
    with ctx:
        try:
            driver.run(script)
            # Retrieve relaxed atoms
            return driver.get_atoms(elements)
        finally:
            driver.lmp.close()
