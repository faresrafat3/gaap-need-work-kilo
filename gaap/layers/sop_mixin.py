"""
SOP Execution Mixin

Provides SOP (Standard Operating Procedure) awareness to execution layers.
Integrates with Layer3_Execution to enforce role-based workflows.

Features:
- Automatic role detection from task type
- SOP step tracking during execution
- Artifact validation before task completion
- Reflexion trigger for skipped steps
"""

import logging
from typing import Any

from gaap.core.types import TaskType
from gaap.core.governance import (
    SOPGatekeeper,
    SOPStore,
    SOPExecution,
    SOPStepStatus,
    ArtifactStatus,
)


class SOPExecutionMixin:
    """
    Mixin for SOP-aware execution.

    Provides methods for:
    - Detecting the appropriate SOP role from task metadata
    - Tracking SOP steps during execution
    - Validating mandatory artifacts
    - Generating reflexion prompts for deviations

    Usage:
        class Layer3Execution(BaseLayer, SOPExecutionMixin):
            def __init__(self, ...):
                self._init_sop(sop_enabled=True)

            async def process(self, task):
                # Start SOP tracking
                self._start_sop_tracking(task)

                # Execute with step tracking
                for step in self._get_pending_steps():
                    result = await self._execute_step(step)
                    if result.success:
                        self._complete_step(step.step_id, result.output)
                    else:
                        self._skip_step(step.step_id, result.error)

                # Validate completion
                if not self._validate_sop_completion(task.id):
                    # Generate reflexion
                    return self._create_reflexion_response(task)
    """

    ROLE_MAPPING: dict[TaskType, str] = {
        TaskType.CODE_GENERATION: "coder",
        TaskType.CODE_REVIEW: "critic",
        TaskType.RESEARCH: "researcher",
        TaskType.ARCHITECTURE: "architect",
        TaskType.TESTING: "coder",
        TaskType.DOCUMENTATION: "researcher",
        TaskType.DEBUGGING: "critic",
        TaskType.REFACTORING: "coder",
        TaskType.ANALYSIS: "researcher",
        TaskType.PLANNING: "architect",
    }

    def _init_sop(
        self,
        sop_enabled: bool = True,
        sop_store: SOPStore | None = None,
    ) -> None:
        """
        Initialize SOP tracking.

        Args:
            sop_enabled: Whether SOP enforcement is enabled
            sop_store: Optional custom SOP store
        """
        self._sop_enabled = sop_enabled
        self._sop_store = sop_store or SOPStore()
        self._sop_gatekeeper = SOPGatekeeper(sop_store=self._sop_store)
        self._current_sop_execution: SOPExecution | None = None
        self._sop_logger = logging.getLogger("gaap.sop.execution")

    def _detect_role(self, task: Any) -> str:
        """
        Detect the appropriate SOP role from task metadata.

        Args:
            task: The task being executed

        Returns:
            Role ID (coder, critic, researcher, architect)
        """
        # Check if task has explicit role
        if hasattr(task, "sop_role") and task.sop_role:
            return task.sop_role

        # Check if task has explicit type
        if hasattr(task, "type") and task.type:
            role = self.ROLE_MAPPING.get(task.type)
            if role:
                return role

        # Check task name/description for hints
        task_name = getattr(task, "name", "").lower()
        task_desc = getattr(task, "description", "").lower()

        if any(kw in task_name or kw in task_desc for kw in ["review", "audit", "security"]):
            return "critic"
        if any(kw in task_name or kw in task_desc for kw in ["research", "investigate", "analyze"]):
            return "researcher"
        if any(kw in task_name or kw in task_desc for kw in ["design", "architecture", "system"]):
            return "architect"

        # Default to coder
        return "coder"

    def _start_sop_tracking(self, task: Any) -> SOPExecution | None:
        """
        Start SOP execution tracking for a task.

        Args:
            task: The task being executed

        Returns:
            SOPExecution instance or None if SOP disabled
        """
        if not self._sop_enabled:
            return None

        role_id = self._detect_role(task)
        task_id = getattr(task, "id", "unknown")

        execution = self._sop_gatekeeper.start_execution(role_id, task_id)

        if execution:
            self._current_sop_execution = execution
            self._sop_logger.info(
                f"Started SOP tracking for task {task_id} with role {role_id} "
                f"({len(execution.role.sop_steps)} steps)"
            )
        else:
            self._sop_logger.warning(f"Failed to start SOP tracking for role {role_id}")

        return execution

    def _get_pending_steps(self) -> list[Any]:
        """
        Get list of pending SOP steps.

        Returns:
            List of pending SOPStep objects
        """
        if not self._current_sop_execution:
            return []

        steps = []
        for step in self._current_sop_execution.role.sop_steps:
            if step.status == SOPStepStatus.PENDING:
                steps.append(step)

        return steps

    def _get_current_step(self) -> Any | None:
        """
        Get the current SOP step.

        Returns:
            Current SOPStep or None
        """
        if not self._current_sop_execution:
            return None

        return self._current_sop_execution.role.get_next_pending_step()

    def _complete_step(
        self,
        step_id: int,
        output: str | None = None,
    ) -> bool:
        """
        Mark an SOP step as completed.

        Args:
            step_id: The step ID
            output: Optional output from the step

        Returns:
            True if step was completed successfully
        """
        if not self._current_sop_execution:
            return False

        task_id = self._current_sop_execution.task_id
        success = self._sop_gatekeeper.complete_step(task_id, step_id, output)

        if success:
            self._sop_logger.debug(f"Completed SOP step {step_id} for task {task_id}")

        return success

    def _skip_step(
        self,
        step_id: int,
        reason: str,
    ) -> bool:
        """
        Mark an SOP step as skipped.

        This triggers reflexion requirement.

        Args:
            step_id: The step ID
            reason: Why the step was skipped

        Returns:
            True if step was skipped successfully
        """
        if not self._current_sop_execution:
            return False

        task_id = self._current_sop_execution.task_id
        success = self._sop_gatekeeper.skip_step(task_id, step_id, reason)

        if success:
            self._sop_logger.warning(f"Skipped SOP step {step_id} for task {task_id}: {reason}")

        return success

    def _register_artifact(
        self,
        artifact_name: str,
        content: Any,
    ) -> bool:
        """
        Register an artifact for SOP validation.

        Args:
            artifact_name: Name of the artifact
            content: The artifact content

        Returns:
            True if artifact was validated successfully
        """
        if not self._current_sop_execution:
            return True  # No SOP tracking, accept artifact

        task_id = self._current_sop_execution.task_id
        is_valid = self._sop_gatekeeper.validate_artifact(task_id, artifact_name, content)

        if is_valid:
            self._sop_logger.info(f"Artifact '{artifact_name}' validated for task {task_id}")
        else:
            self._sop_logger.warning(f"Artifact '{artifact_name}' invalid for task {task_id}")

        return is_valid

    def _validate_sop_completion(self, task_id: str | None = None) -> dict[str, Any]:
        """
        Validate that SOP requirements are met.

        Args:
            task_id: Optional task ID (uses current execution if not provided)

        Returns:
            Dict with completion status and details
        """
        if not self._sop_enabled or not self._current_sop_execution:
            return {"complete": True, "reason": "SOP tracking disabled"}

        actual_task_id = task_id or self._current_sop_execution.task_id
        result = self._sop_gatekeeper.check_completion(actual_task_id)

        if not result["complete"]:
            self._sop_logger.warning(
                f"SOP not complete for task {actual_task_id}: "
                f"steps={result['steps_completion_rate']:.0%}, "
                f"missing_artifacts={result['missing_artifacts']}, "
                f"skipped_steps={result['skipped_steps']}"
            )

        return result

    def _get_reflexion_prompt(self) -> str | None:
        """
        Get reflexion prompt for skipped steps.

        Returns:
            Reflexion prompt string or None
        """
        if not self._current_sop_execution:
            return None

        task_id = self._current_sop_execution.task_id
        return self._sop_gatekeeper.get_reflexion_prompt(task_id)

    def _requires_reflexion(self) -> bool:
        """
        Check if reflexion is required for current execution.

        Returns:
            True if reflexion is required
        """
        if not self._current_sop_execution:
            return False

        return self._current_sop_execution.reflexion_required

    def _get_sop_summary(self) -> dict[str, Any]:
        """
        Get summary of SOP execution.

        Returns:
            Dict with SOP execution summary
        """
        if not self._current_sop_execution:
            return {"sop_enabled": False}

        execution = self._current_sop_execution
        completion = self._validate_sop_completion()

        return {
            "sop_enabled": True,
            "execution_id": execution.execution_id,
            "role": execution.role.name,
            "task_id": execution.task_id,
            "current_step": execution.current_step,
            "total_steps": len(execution.role.sop_steps),
            "completion_rate": completion.get("steps_completion_rate", 0),
            "skipped_steps": execution.skipped_steps,
            "artifacts_produced": list(execution.artifacts_produced.keys()),
            "missing_artifacts": completion.get("missing_artifacts", []),
            "reflexion_required": execution.reflexion_required,
        }

    def _end_sop_tracking(self) -> dict[str, Any]:
        """
        End SOP tracking and return final summary.

        Returns:
            Final SOP execution summary
        """
        summary = self._get_sop_summary()
        self._current_sop_execution = None
        return summary

    def _get_available_roles(self) -> list[str]:
        """
        Get list of available SOP roles.

        Returns:
            List of role IDs
        """
        return self._sop_store.list_roles()

    def _get_role_info(self, role_id: str) -> dict[str, Any] | None:
        """
        Get information about a specific role.

        Args:
            role_id: The role ID

        Returns:
            Dict with role information or None
        """
        role = self._sop_store.get_role(role_id)
        if not role:
            return None

        return {
            "role_id": role.role_id,
            "name": role.name,
            "mission": role.mission,
            "steps": [{"id": s.step_id, "description": s.description} for s in role.sop_steps],
            "mandatory_artifacts": [a.name for a in role.mandatory_artifacts],
        }
