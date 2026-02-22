"""
Tool Synthesizer (Just-in-Time Tooling)
New Module - Evolution 2026
Implements: docs/evolution_plan_2026/14_JUST_IN_TIME_TOOLING.md
"""

import logging
import ast
import uuid
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

# Import security modules we built earlier
from gaap.security.dlp import DLPScanner
from gaap.core.axioms import AxiomValidator

logger = logging.getLogger("gaap.tools.synthesizer")


@dataclass
class SynthesizedTool:
    id: str
    name: str
    code: str
    description: str
    file_path: Path
    is_safe: bool
    module: Any = None


class ToolSynthesizer:
    """
    Generates, validates, and hot-loads new Python tools on the fly.
    """

    def __init__(self, workspace_path: Path = Path(".gaap/custom_tools")):
        self.workspace_path = workspace_path
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Add workspace to python path so we can import generated modules
        if str(self.workspace_path.absolute()) not in sys.path:
            sys.path.append(str(self.workspace_path.absolute()))

        self.axiom_validator = AxiomValidator()
        self.dlp = DLPScanner()

    def __repr__(self) -> str:
        return f"ToolSynthesizer(workspace={self.workspace_path})"

    async def synthesize(self, intent: str, code_content: str) -> Optional[SynthesizedTool]:
        """
        Takes raw code, validates it, saves it, and loads it as a callable tool.
        """
        logger.info(f"Synthesizing tool for intent: {intent}")

        # 1. Validation Phase (The Filter)
        axiom_results = self.axiom_validator.validate(code_content)
        failed_axioms = [r for r in axiom_results if not r.passed]

        if failed_axioms:
            logger.error(
                f"Tool synthesis rejected by Constitution: {[r.axiom_name for r in failed_axioms]}"
            )
            return None

        # 2. Syntax Check
        try:
            ast.parse(code_content)
        except SyntaxError as e:
            logger.error(f"Generated tool has syntax error: {e}")
            return None

        # 3. Naming & Storage
        tool_id = str(uuid.uuid4())[:8]
        tool_name = f"tool_{tool_id}"
        file_path = self.workspace_path / f"{tool_name}.py"

        # Write to disk
        with open(file_path, "w") as f:
            f.write(code_content)

        # 4. Hot-Loading (The Magic)
        try:
            spec = importlib.util.spec_from_file_location(tool_name, file_path)
            if not spec or not spec.loader:
                raise ImportError("Failed to create module spec")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify the module has a 'run' or 'execute' function
            if not hasattr(module, "run") and not hasattr(module, "execute"):
                logger.warning(f"Tool {tool_name} has no entry point (run/execute).")

            return SynthesizedTool(
                id=tool_id,
                name=tool_name,
                code=code_content,
                description=intent,
                file_path=file_path,
                is_safe=True,
                module=module,
            )

        except Exception as e:
            logger.error(f"Failed to hot-load tool: {e}")
            return None

    def cleanup(self):
        """Removes temporary tools."""
        # Implementation to delete files later
        pass
