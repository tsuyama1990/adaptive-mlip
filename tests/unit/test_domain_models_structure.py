import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_structure_config_valid_default() -> None:
    config = StructureConfig(elements=["Fe", "Pt"], supercell_size=[2, 2, 2])
    assert config.policy_name == ExplorationPolicy.COLD_START
    assert config.rattle_stdev == 0.1
    assert config.strain_mode == "full"
    assert config.vacancy_rate == 0.05


def test_structure_config_valid_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.2,
    )
    assert config.policy_name == ExplorationPolicy.RANDOM_RATTLE
    assert config.rattle_stdev == 0.2


def test_structure_config_invalid_rattle_stdev() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1], rattle_stdev=-0.1)


def test_structure_config_invalid_vacancy_rate() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1], vacancy_rate=1.1)


def test_structure_config_invalid_policy() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1], policy_name="invalid_policy")
