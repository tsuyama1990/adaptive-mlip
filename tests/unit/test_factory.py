from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.domain_models import PyAceConfig, WorkflowConfig
from pyacemaker.factory import ModuleFactory
from pyacemaker.structure_generator.direct import DirectSampler
from tests.conftest import create_dummy_pseudopotentials


@pytest.fixture
def mock_config(
    mock_structure_config: Any,
    mock_dft_config: Any,
    mock_training_config: Any,
    mock_md_config: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> PyAceConfig:
    monkeypatch.chdir(tmp_path)
    # Create required pseudo file used in shared fixture
    create_dummy_pseudopotentials(tmp_path, ["Fe"])

    # Update DFT config to use the created file and match element
    mock_dft_config.pseudopotentials = {"Fe": "Fe.UPF"}

    return PyAceConfig(
        project_name="TestFactory",
        structure=mock_structure_config,
        dft=mock_dft_config,
        training=mock_training_config,
        md=mock_md_config,
        workflow=WorkflowConfig(max_iterations=1),
    )


def test_module_factory_create_modules(mock_config: PyAceConfig) -> None:
    """Test that factory creates correct module instances."""

    # We patch DFTManager to avoid QEDriver init (which checks pseudo existence)
    # Actually mock_config fixture creates the pseudo file, so DFTManager init is safe.
    # But QEDriver might check other things.
    # Let's patch DFTManager anyway to isolate Factory test.
    with patch("pyacemaker.factory.DFTManager") as MockDFTManager:
        gen, oracle, trainer, engine, active_set, validator = ModuleFactory.create_modules(mock_config)

        assert isinstance(gen, DirectSampler)
        assert isinstance(trainer, PacemakerTrainer)
        assert isinstance(engine, LammpsEngine)
        assert active_set is not None

        # Oracle should be what DFTManager returns
        assert oracle == MockDFTManager.return_value

        # Verify initializations
        MockDFTManager.assert_called_once_with(mock_config.dft)

        # Check trainer config
        assert trainer.config == mock_config.training

        # Check engine config
        assert engine.config == mock_config.md


def test_module_factory_invalid_config() -> None:
    """Test factory validation."""
    config = MagicMock(spec=PyAceConfig)
    config.project_name = ""

    with pytest.raises(ConfigError, match="Project name is required"):
        ModuleFactory.create_modules(config)
