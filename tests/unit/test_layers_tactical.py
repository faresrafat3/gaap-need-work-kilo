"""
Enhanced tests for Layer 2 - Tactical Layer
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.layers.layer2_tactical import (
    TaskCategory,
    DependencyType,
    AtomicTask,
    TaskNode,
    TaskGraph,
    DependencyResolver,
    TacticalDecomposer,
    ExecutionQueue,
    Layer2Tactical,
)


class TestTaskCategory:
    """Tests for TaskCategory enum"""

    def test_task_category_values(self):
        """Test all task category values"""
        assert TaskCategory.SETUP is not None
        assert TaskCategory.DATABASE is not None
        assert TaskCategory.API is not None
        assert TaskCategory.FRONTEND is not None
        assert TaskCategory.TESTING is not None
        assert TaskCategory.DOCUMENTATION is not None
        assert TaskCategory.INTEGRATION is not None
        assert TaskCategory.SECURITY is not None
        assert TaskCategory.INFRASTRUCTURE is not None

    def test_task_category_research(self):
        """Test research categories"""
        assert TaskCategory.INFORMATION_GATHERING is not None
        assert TaskCategory.SOURCE_VERIFICATION is not None
        assert TaskCategory.DATA_SYNTHESIS is not None

    def test_task_category_diagnostics(self):
        """Test diagnostic categories"""
        assert TaskCategory.REPRODUCTION is not None
        assert TaskCategory.LOG_ANALYSIS is not None
        assert TaskCategory.ROOT_CAUSE_ANALYSIS is not None
        assert TaskCategory.MITIGATION is not None


class TestDependencyType:
    """Tests for DependencyType enum"""

    def test_dependency_type_values(self):
        """Test dependency type values"""
        assert DependencyType.HARD.value == "hard"
        assert DependencyType.SOFT.value == "soft"
        assert DependencyType.CONDITIONAL.value == "conditional"


class TestAtomicTask:
    """Tests for AtomicTask dataclass"""

    def test_create_atomic_task(self):
        """Test creating an atomic task"""
        task = AtomicTask(
            id="atomic-001",
            name="Write function",
            description="Write a Python function",
            category=TaskCategory.API,
        )
        assert task.id == "atomic-001"
        assert task.category == TaskCategory.API

    def test_atomic_task_defaults(self):
        """Test atomic task default values"""
        task = AtomicTask(
            id="test",
            name="test",
            description="test",
            category=TaskCategory.API,
        )
        assert task.estimated_tokens == 500
        assert task.estimated_time_minutes == 5
        assert task.estimated_cost_usd == 0.01
        assert task.retry_count == 0

    def test_atomic_task_to_task(self):
        """Test converting atomic task to Task"""
        from gaap.core.types import TaskPriority, TaskComplexity, TaskType

        atomic = AtomicTask(
            id="atomic-001",
            name="Write function",
            description="Write a Python function",
            category=TaskCategory.API,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
        )
        task = atomic.to_task()

        assert task.id == "atomic-001"


class TestTaskNode:
    """Tests for TaskNode"""

    def test_create_task_node(self):
        """Test creating a task node"""
        task = AtomicTask(
            id="task-001",
            name="Test Task",
            description="Test description",
            category=TaskCategory.API,
        )
        node = TaskNode(task=task)
        assert node.id == "task-001"

    def test_task_node_is_ready(self):
        """Test node is_ready method"""
        task = AtomicTask(
            id="task-1",
            name="T1",
            description="t1",
            category=TaskCategory.API,
            dependencies=["dep-1"],
        )
        node = TaskNode(task=task)
        assert node.is_ready(completed={"dep-1"}) is True
        assert node.is_ready(completed=set()) is False


class TestTaskGraph:
    """Tests for TaskGraph"""

    def test_create_task_graph(self):
        """Test creating a task graph"""
        graph = TaskGraph()
        assert graph is not None

    def test_add_task(self):
        """Test adding tasks to graph"""
        graph = TaskGraph()
        task = AtomicTask(
            id="task-1",
            name="Task 1",
            description="First task",
            category=TaskCategory.API,
        )
        node = graph.add_task(task)

        assert len(graph.all_nodes) == 1
        assert "task-1" in graph.all_nodes
        assert graph.total_tasks == 1

    def test_add_dependency(self):
        """Test adding dependencies"""
        graph = TaskGraph()
        task1 = AtomicTask(id="task-1", name="T1", description="t1", category=TaskCategory.API)
        task2 = AtomicTask(
            id="task-2",
            name="T2",
            description="t2",
            category=TaskCategory.API,
            dependencies=["task-1"],
        )

        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_dependency("task-2", "task-1", DependencyType.HARD)

        assert len(graph.all_nodes["task-2"].parents) == 1

    def test_get_ready_tasks(self):
        """Test getting ready tasks"""
        graph = TaskGraph()
        task1 = AtomicTask(id="task-1", name="T1", description="t1", category=TaskCategory.API)
        task2 = AtomicTask(
            id="task-2",
            name="T2",
            description="t2",
            category=TaskCategory.API,
            dependencies=["task-1"],
        )

        graph.add_task(task1)
        graph.add_task(task2)

        ready = graph.get_ready_tasks(completed=set(), in_progress=set())
        assert len(ready) == 1
        assert ready[0].id == "task-1"

    def test_get_ready_with_completed(self):
        """Test getting ready tasks with completed"""
        graph = TaskGraph()
        task1 = AtomicTask(id="task-1", name="T1", description="t1", category=TaskCategory.API)
        task2 = AtomicTask(
            id="task-2",
            name="T2",
            description="t2",
            category=TaskCategory.API,
            dependencies=["task-1"],
        )

        graph.add_task(task1)
        graph.add_task(task2)

        ready = graph.get_ready_tasks(completed={"task-1"}, in_progress=set())
        assert len(ready) == 1
        assert ready[0].id == "task-2"


class TestDependencyResolver:
    """Tests for DependencyResolver"""

    def test_create_resolver(self):
        """Test creating a dependency resolver"""
        resolver = DependencyResolver()
        assert resolver is not None


class TestTacticalDecomposer:
    """Tests for TacticalDecomposer"""

    def test_create_decomposer(self):
        """Test creating a tactical decomposer"""
        decomposer = TacticalDecomposer()
        assert decomposer is not None


class TestExecutionQueue:
    """Tests for ExecutionQueue"""

    def test_create_queue(self):
        """Test creating an execution queue"""
        queue = ExecutionQueue(max_parallel=10)
        assert queue is not None

    def test_enqueue(self):
        """Test enqueuing tasks"""
        queue = ExecutionQueue(max_parallel=10)
        task1 = AtomicTask(id="task-1", name="T1", description="t1", category=TaskCategory.API)
        task2 = AtomicTask(id="task-2", name="T2", description="t2", category=TaskCategory.API)
        queue.enqueue([task1, task2])

        assert len(queue._queue) == 2

    def test_get_progress(self):
        """Test getting progress"""
        queue = ExecutionQueue(max_parallel=10)
        progress = queue.get_progress()

        assert "total" in progress
        assert "completed" in progress
        assert "in_progress" in progress


class TestLayer2Tactical:
    """Tests for Layer 2 Tactical"""

    def test_layer2_initialization(self):
        """Test Layer 2 initialization"""
        layer2 = Layer2Tactical()
        assert layer2 is not None

    @pytest.mark.asyncio
    async def test_layer2_process(self):
        """Test Layer 2 processing"""
        from gaap.layers.layer1_strategic import ArchitectureSpec

        layer2 = Layer2Tactical()
        spec = ArchitectureSpec(spec_id="test")

        result = await layer2.process(spec)
        assert result is not None

    def test_layer2_stats(self):
        """Test Layer 2 statistics"""
        layer2 = Layer2Tactical()
        assert layer2._tasks_created == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
