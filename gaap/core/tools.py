import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.tools")


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable


class ToolRegistry:
    """Registry for GAAP tools"""

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register basic file system and system tools"""
        self.register(
            "list_dir",
            "Lists files and directories in a given path.",
            {"path": "string"},
            self.list_dir,
        )
        self.register(
            "read_file", "Reads the content of a file.", {"path": "string"}, self.read_file
        )
        self.register(
            "write_file",
            "Writes content to a file. Use with caution.",
            {"path": "string", "content": "string"},
            self.write_file,
        )
        self.register(
            "run_command",
            "Runs a shell command in the workspace.",
            {"command": "string"},
            self.run_command,
        )

    def register(self, name: str, description: str, parameters: dict[str, Any], func: Callable):
        self.tools[name] = ToolDefinition(name, description, parameters, func)

    def _safe_path(self, path_str: str) -> Path:
        """Ensure the path is within the workspace root for security."""
        target = (self.workspace_root / path_str).resolve()
        if not str(target).startswith(str(self.workspace_root)):
            raise PermissionError(f"Access denied: Path {path_str} is outside workspace.")
        return target

    # --- Tool Implementations ---

    def list_dir(self, path: str = ".") -> str:
        try:
            target = self._safe_path(path)
            items = os.listdir(target)
            return "\n".join(
                [f"{'[DIR] ' if os.path.isdir(target/i) else '      '}{i}" for i in items]
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def read_file(self, path: str) -> str:
        try:
            target = self._safe_path(path)
            return target.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_command(self, command: str) -> str:
        try:
            # Simple restricted command runner
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.workspace_root),
                timeout=30,
            )
            return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool by name with parameters"""
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."

        logger.info(f"Executing tool: {name} with args: {kwargs}")
        return self.tools[name].func(**kwargs)

    def get_instructions(self) -> str:
        """Returns a STRICT One-Shot instruction block for the LLM."""
        instr = "\n" + "#" * 60 + "\n"
        instr += "🔴 SYSTEM ROLE: AUTONOMOUS SYSTEM OPERATOR 🔴\n"
        instr += "You are NOT a chat assistant. You are a SYSTEM OPERATOR.\n"
        instr += "Your goal is to execute tasks by calling TOOLS.\n"
        instr += "DO NOT output Python code blocks for the user to run. EXECUTE actions yourself using TOOLS.\n\n"

        instr += "### MANDATORY INTERACTION PROTOCOL ###\n"
        instr += "1. Write THOUGHT: <why you are calling this tool>\n"
        instr += "2. Write CALL: tool_name(parameter='value', ...)\n"
        instr += "3. Wait for the TOOL RESULT from the system.\n"
        instr += "4. Provide FINAL_ANSWER only when the task is fully complete.\n\n"

        instr += "### EXAMPLE INTERACTION ###\n"
        instr += "User: Create a file named 'hello.txt' with content 'hi'.\n"
        instr += "Assistant:\n"
        instr += "THOUGHT: I will use the write_file tool to create the file.\n"
        instr += "CALL: write_file(path='hello.txt', content='hi')\n"
        instr += "System: TOOL RESULT (write_file): Successfully wrote to hello.txt\n"
        instr += "Assistant:\n"
        instr += "FINAL_ANSWER: I have created the file as requested.\n\n"

        instr += "### AVAILABLE TOOLS ###\n"
        for t in self.tools.values():
            instr += f"- {t.name}: {t.description}\n  Usage: CALL: {t.name}({', '.join([f'{k}=\'...\"' for k in t.parameters])})\n"
        instr += "#" * 60 + "\n"
        return instr
