# TECHNICAL SPECIFICATION: Interface Evolution (Rich TUI & Live Feedback)

**Target:** `gaap/cli/`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Blocking Execution:** No live feedback during deep thought cycles.
- **Plain-Text Output:** Hard to parse complex architectural specs or code diffs.
- **Disconnected UX:** User cannot "steer" the agent mid-task.

## 2. Refactoring Requirements

### 2.1 Implementing Rich Live-Stream
Integrate the `rich.live` and `rich.progress` modules.
- **Requirement:** Display a "Brain Activity" spinner with the current sub-task name.
- **Requirement:** Stream the LLM response word-by-word into a syntax-highlighted Markdown panel.

### 2.2 Terminal User Interface (TUI) Mode
Add `gaap dashboard` command using the `Textual` framework.
- **Layout:**
    - *Top:* OODA Loop status (Observe/Orient/Decide/Act).
    - *Left:* Current Task Graph (DAG) with progress bars.
    - *Center:* Real-time Logs / Code generation.
    - *Right:* Real-time Budget & Token monitor.

### 2.3 Interactive Interventions (The "Pause" Pattern)
Allow the user to interrupt the loop without killing the process.
- **Action:** Press `Ctrl+C` once during a task to enter **Steering Mode**.
- **UX:** "Task paused. Enter adjustment (e.g. 'Actually use FastAPI instead of Flask') or 'resume'."

### 2.4 Pretty Diffs & Artifact Previews
Use `rich.syntax` and `rich.panel`.
- **Logic:** Before writing a file, show a side-by-side diff.
- **Requirement:** Color-code changes (Green for add, Red for delete).

## 3. Implementation Steps
1.  **Add** `rich` and `textual` to dependencies.
2.  **Refactor** `gaap/cli/commands/cmd_chat.py` to use `Live` panels.
3.  **Create** a `TUIApp` class in `gaap/cli/tui.py`.

---
**Handover Status:** Ready. Code Agent must implement 'Rich Live Panels' for the next demo to improve UX.
