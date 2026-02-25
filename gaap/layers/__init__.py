"""
GAAP 4-Layer Architecture Module

Provides the core 4-layer cognitive architecture:

Layers:
    - Layer 0 (Interface): Security, intent classification, routing
    - Layer 1 (Strategic): Tree of Thoughts, MAD Architecture Panel
    - Layer 2 (Tactical): Task decomposition, DAG construction
    - Layer 3 (Execution): Parallel execution, quality assurance

Usage:
    from gaap.layers import (
        Layer0Interface,
        Layer1Strategic,
        Layer2Tactical,
        Layer3Execution
    )

    # Create layers
    l0 = Layer0Interface()
    l1 = Layer1Strategic()
    l2 = Layer2Tactical()
    l3 = Layer3Execution()
"""

from .layer0_interface import (
    IntentType,
    Layer0Interface,
    StructuredIntent,
)
from .layer1_strategic import (
    ArchitectureParadigm,
    ArchitectureSpec,
    Layer1Strategic,
    StrategicPlanner,
    ToTStrategic,
)
from .layer2_tactical import (
    AtomicTask,
    DependencyResolver,
    Layer2Tactical,
    TacticalDecomposer,
    TaskGraph,
)
from .layer3_execution import (
    ExecutionResult,
    ExecutorPool,
    GeneticTwin,
    Layer3Execution,
    MADQualityPanel,
    QualityPipeline,
)
from .mcts_logic import (
    MCTSConfig,
    MCTSNode,
    MCTSPhase,
    MCTSStrategic,
    NodeType,
    ValueOracle,
    create_mcts,
    create_mcts_for_priority,
)
from .sop_manager import (
    ArtifactValidationResult,
    QualityGate,
    QualityGateStatus,
    SOP,
    SOPManager,
    SOPStep,
    StepType,
    create_sop_manager,
)

__all__ = [
    "Layer0Interface",
    "IntentType",
    "StructuredIntent",
    # Layer 1
    "Layer1Strategic",
    "StrategicPlanner",
    "ToTStrategic",
    "ArchitectureParadigm",
    "ArchitectureSpec",
    # MCTS
    "MCTSStrategic",
    "MCTSConfig",
    "MCTSNode",
    "MCTSPhase",
    "NodeType",
    "ValueOracle",
    "create_mcts",
    "create_mcts_for_priority",
    # Layer 2
    "Layer2Tactical",
    "TacticalDecomposer",
    "DependencyResolver",
    "AtomicTask",
    "TaskGraph",
    # Layer 3
    "Layer3Execution",
    "ExecutorPool",
    "GeneticTwin",
    "ExecutionResult",
    "MADQualityPanel",
    "QualityPipeline",
    # SOP Manager
    "SOPManager",
    "SOP",
    "SOPStep",
    "StepType",
    "QualityGate",
    "QualityGateStatus",
    "ArtifactValidationResult",
    "create_sop_manager",
]
