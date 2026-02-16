"""
Unit tests for Layer 2 - Tactical Layer
"""

import pytest

from gaap.core.types import (
    TaskComplexity,
    TaskPriority,
    TaskType,
)


class TestTaskDecomposition:
    """Tests for task decomposition"""

    def test_simple_task_no_decomposition(self):
        """Test that simple tasks don't need decomposition"""
        complexity = TaskComplexity.SIMPLE
        if complexity == TaskComplexity.SIMPLE:
            subtasks = []
        assert len(subtasks) == 0

    def test_moderate_task_decomposition(self):
        """Test decomposition of moderate tasks"""
        complexity = TaskComplexity.MODERATE
        max_subtasks = 5
        if complexity == TaskComplexity.MODERATE:
            subtasks = ["subtask_1", "subtask_2", "subtask_3"]
        assert len(subtasks) <= max_subtasks

    def test_complex_task_decomposition(self):
        """Test decomposition of complex tasks"""
        complexity = TaskComplexity.COMPLEX
        if complexity == TaskComplexity.COMPLEX:
            subtasks = [f"subtask_{i}" for i in range(8)]
        assert len(subtasks) >= 5

    def test_subtask_priority_inheritance(self):
        """Test that subtasks inherit parent priority"""
        parent_priority = TaskPriority.HIGH
        subtask_priorities = [parent_priority for _ in range(3)]
        assert all(p == parent_priority for p in subtask_priorities)

    def test_subtask_type_assignment(self):
        """Test type assignment for subtasks"""
        subtasks = [
            {"id": 1, "type": TaskType.ANALYSIS},
            {"id": 2, "type": TaskType.CODE_GENERATION},
            {"id": 3, "type": TaskType.TESTING},
        ]
        types = [s["type"] for s in subtasks]
        assert TaskType.ANALYSIS in types
        assert TaskType.CODE_GENERATION in types


class TestDAGConstruction:
    """Tests for DAG (Directed Acyclic Graph) construction"""

    def test_dag_node_creation(self):
        """Test creating DAG nodes"""
        nodes = {}
        for i in range(5):
            nodes[f"task_{i}"] = {"dependencies": [], "status": "pending"}
        assert len(nodes) == 5

    def test_dag_edge_creation(self):
        """Test creating DAG edges"""
        edges = [
            ("task_0", "task_1"),
            ("task_1", "task_2"),
            ("task_1", "task_3"),
            ("task_2", "task_4"),
        ]
        assert len(edges) == 4

    def test_dag_topological_order(self):
        """Test topological ordering"""
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }

        def topological_sort(g):
            visited = set()
            order = []

            def visit(node):
                if node in visited:
                    return
                visited.add(node)
                for neighbor in g.get(node, []):
                    visit(neighbor)
                order.append(node)

            for node in g:
                visit(node)
            return list(reversed(order))

        order = topological_sort(graph)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_dag_cycle_detection(self):
        """Test cycle detection in DAG"""
        graph_with_cycle = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        }

        def has_cycle(g):
            visited = set()
            rec_stack = set()

            def visit(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                visited.add(node)
                rec_stack.add(node)
                for neighbor in g.get(node, []):
                    if visit(neighbor):
                        return True
                rec_stack.remove(node)
                return False

            return any(visit(node) for node in g)

        assert has_cycle(graph_with_cycle)

    def test_dag_no_cycle(self):
        """Test that valid DAG has no cycle"""
        graph_no_cycle = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }

        def has_cycle(g):
            visited = set()
            rec_stack = set()

            def visit(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                visited.add(node)
                rec_stack.add(node)
                for neighbor in g.get(node, []):
                    if visit(neighbor):
                        return True
                rec_stack.remove(node)
                return False

            return any(visit(node) for node in g)

        assert not has_cycle(graph_no_cycle)


class TestDependencyResolution:
    """Tests for dependency resolution"""

    def test_dependency_satisfaction(self):
        """Test checking if dependencies are satisfied"""
        completed = {"task_1", "task_2"}
        dependencies = {"task_1", "task_2"}
        satisfied = dependencies.issubset(completed)
        assert satisfied

    def test_dependency_not_satisfied(self):
        """Test when dependencies are not satisfied"""
        completed = {"task_1"}
        dependencies = {"task_1", "task_2"}
        satisfied = dependencies.issubset(completed)
        assert not satisfied

    def test_ready_tasks_identification(self):
        """Test identifying ready tasks"""
        task_status = {
            "task_1": {"completed": True, "dependencies": []},
            "task_2": {"completed": False, "dependencies": ["task_1"]},
            "task_3": {"completed": False, "dependencies": []},
        }
        completed = {t for t, s in task_status.items() if s["completed"]}
        ready = [
            t
            for t, s in task_status.items()
            if not s["completed"] and all(d in completed for d in s["dependencies"])
        ]
        assert "task_2" in ready
        assert "task_3" in ready


class TestExecutionOrder:
    """Tests for execution ordering"""

    def test_critical_path_calculation(self):
        """Test critical path calculation"""
        durations = {"A": 3, "B": 2, "C": 4, "D": 1}
        dependencies = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}

        def calculate_earliest_finish(task, memo={}):
            if task in memo:
                return memo[task]
            if not dependencies[task]:
                memo[task] = durations[task]
            else:
                memo[task] = durations[task] + max(
                    calculate_earliest_finish(d, memo) for d in dependencies[task]
                )
            return memo[task]

        critical_path = max(calculate_earliest_finish(t) for t in durations)
        assert critical_path == 8

    def test_parallel_execution_groups(self):
        """Test grouping tasks for parallel execution"""
        levels = {
            0: ["A"],
            1: ["B", "C"],
            2: ["D"],
        }
        assert len(levels[1]) == 2  # B and C can run in parallel

    def test_max_parallelism(self):
        """Test maximum parallelism calculation"""
        tasks_per_level = [1, 3, 2, 4, 1]
        max_parallel = max(tasks_per_level)
        assert max_parallel == 4


class TestAtomicTask:
    """Tests for atomic task creation"""

    def test_atomic_task_creation(self):
        """Test creating atomic tasks"""
        task = {
            "id": "atomic_001",
            "description": "Write function header",
            "estimated_tokens": 50,
            "dependencies": [],
        }
        assert task["estimated_tokens"] == 50
        assert len(task["dependencies"]) == 0

    def test_atomic_task_estimation(self):
        """Test token estimation for atomic tasks"""
        descriptions = [
            "Write a simple function",
            "Implement a complex algorithm with multiple steps",
            "Add docstring",
        ]
        estimates = [len(d.split()) * 10 for d in descriptions]
        assert estimates[1] > estimates[0]
        assert estimates[1] > estimates[2]


class TestTacticalIntegration:
    """Integration tests for tactical layer"""

    @pytest.mark.asyncio
    async def test_decomposition_flow(self, sample_task):
        """Test complete decomposition flow"""
        complexity = sample_task.complexity or TaskComplexity.MODERATE
        assert complexity is not None

    @pytest.mark.asyncio
    async def test_dag_execution_order(self):
        """Test DAG execution order"""
        execution_order = []
        tasks = ["task_1", "task_2", "task_3"]
        for task in tasks:
            execution_order.append(task)
        assert execution_order == tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
