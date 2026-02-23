from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.exceptions import ConfigError
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.factory import Cycle01Engine, Cycle01Generator, Cycle01Trainer, ModuleFactory


@pytest.fixture
def mock_config() -> PyAceConfig:
    # Minimal config to satisfy Pydantic
    return PyAceConfig.model_validate({
        "project_name": "TestFactory",
        "structure": {"elements": ["Fe"], "supercell_size": [1, 1, 1]},
        "dft": {
            "code": "qe",
            "functional": "PBE",
            "kpoints_density": 0.04,
            "encut": 500.0,
            "pseudopotentials": {"Fe": "Fe.UPF"},
        },
        "training": {"potential_type": "ace", "cutoff_radius": 5.0, "max_basis_size": 500},
        "md": {"temperature": 300, "pressure": 0.0, "timestep": 1.0, "n_steps": 10},
        "workflow": {"max_iterations": 1},
    })


def test_module_factory_create_modules(mock_config: PyAceConfig) -> None:
    """Test that factory creates correct module instances."""

    # We don't want to actually instantiate DFTManager which creates QEDriver
    # So we can patch DFTManager to verify it's called
    with patch("pyacemaker.factory.DFTManager") as MockDFTManager:
        gen, oracle, trainer, engine = ModuleFactory.create_modules(mock_config)

        assert isinstance(gen, Cycle01Generator)
        assert isinstance(trainer, Cycle01Trainer)
        assert isinstance(engine, Cycle01Engine)

        # Oracle should be what DFTManager returns
        assert oracle == MockDFTManager.return_value

        # Verify DFTManager initialized with correct config
        MockDFTManager.assert_called_once_with(mock_config.dft)


def test_module_factory_invalid_config() -> None:
    """Test factory validation."""
    config = MagicMock(spec=PyAceConfig)
    config.project_name = ""

    with pytest.raises(ConfigError, match="Project name is required"):
        ModuleFactory.create_modules(config)
