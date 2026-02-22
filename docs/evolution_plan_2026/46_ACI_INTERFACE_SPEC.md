# TECHNICAL SPECIFICATION: Advanced Agent-Computer Interface (ACI) (v1.0)

**Focus:** Optimizing the terminal and file interactions for LLM efficiency.

## 1. The ACI Philosophy
Standard terminals are designed for human eyes. ACI is designed for LLM "context attention".
- **Density:** More information per token.
- **Guardrails:** Prevents common LLM terminal mistakes (e.g., forgetting to `cd`).

## 2. Component Design

### 2.1 The "Context-Aware" Shell
Instead of raw `bash`, we provide a wrapper tool `gaap_shell`:
- **Auto-PWD:** Every command output starts with the current directory path.
- **Tree-Summary:** After a `cd`, automatically show a depth-2 directory tree.
- **Error Filtering:** If a command fails, use a local SLM to "strip" the noise from the error and send only the relevant lines.

### 2.2 Navigation Tools (Line-Based)
LLMs struggle with large files. We provide "Search-and-View" tools:
- `search_code(pattern)`: Returns line numbers and a 3-line snippet.
- `read_lines(file, start, end)`: Fetches specific chunks to save tokens.
- `apply_diff(patch)`: Enforces atomic file updates instead of rewriting entire files (safer and faster).

## 3. Implementation Logic
1.  **Add** `aci_shell.py` to `gaap/core/tools/`.
2.  **Refactor** `Layer3_Execution` to use `aci_shell` as its primary execution interface.
3.  **Update** System Prompts to teach the LLM how to use these high-density tools.

## 4. Axiom: Feedback Loop
The ACI must provide immediate feedback if an LLM sends a hallucinated flag (e.g., `ls --wrong-flag`). The ACI should reply: "Unknown flag detected. Available flags are: -a, -l...".
