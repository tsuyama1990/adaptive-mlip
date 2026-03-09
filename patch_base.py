<<<<<<< SEARCH
from pyacemaker.domain_models.structure import StructureConfig


class BasePolicy(ABC):
    """
    Abstract base class for exploration policies.
    """

    @abstractmethod
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
=======
from pyacemaker.domain_models.structure import StructureConfig
from pyacemaker.domain_models.workflow import ActiveLearningThresholds, CutoutConfig, LoopStrategyConfig


class BasePolicy(ABC):
    """
    Abstract base class for exploration policies.
    """

    @abstractmethod
    def generate(
        self,
        base_structure: Atoms,
        config: StructureConfig,
        n_structures: int = 1,
        engine: Any | None = None,
        potential: str | Path | None = None,
        thresholds: ActiveLearningThresholds | None = None,
        cutout_config: CutoutConfig | None = None,
        loop_strategy: LoopStrategyConfig | None = None,
    ) -> Iterator[Atoms]:
        """
        Generates new candidates based on policy logic.
        """
>>>>>>> REPLACE
