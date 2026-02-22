# GAAP Evolution: UX/UI Strategic Design Plan

**Focus:** Moving from "Command-and-Control" to "Collaborative Intelligence".

## 1. Design Philosophy: "The Transparent Mind"
GAAP's UX should prioritize **Honesty** over **Politeness**. If the agent is unsure, the UI should reflect that uncertainty (using blur, yellow colors, or "low-confidence" badges).

## 2. Multi-Surface Experience

### 2.1 The Pro-Terminal (CLI)
- **Live Stream:** Use `Rich.Live` to show a spinning brain icon and the current "Micro-Thought".
- **Interactive Prompts:** Instead of simple `input()`, use fuzzy-search menus for choosing providers or tools.
- **Summary Cards:** At the end of a task, show a "Receipt" of work (Time taken, Tokens spent, Files changed, Quality score).

### 2.2 The Holodeck (Web UI)
- **Layer-Based Navigation:** Tabs for "Strategy", "Tactics", and "Execution".
- **The "Dream" Log:** A dedicated view to see what the agent learned during its "Sleep" cycle.
- **Visual Diffing:** A Monaco-based editor showing proposed vs current code.

## 3. Interaction Patterns

### 3.1 The "Pause & Steer" Pattern
- **UI Element:** A global "PAUSE" button.
- **Action:** Freezes the agent's LLM stream.
- **UX:** User can type a "Steering Instruction" (e.g., "Actually, don't use MongoDB, use PostgreSQL") and hit "RESUME". The agent re-calculates Layer 1.

### 3.2 The "Why?" Button
- **UI Element:** A small question mark next to every agent decision.
- **Action:** Opens a sidebar showing:
    - The specific `SemanticRule` used.
    - The top 3 `EpisodicMemories` that influenced this choice.
    - The LLM's internal reasoning string.

## 4. Accessibility & Aesthetics
- **Theme:** "Cyber-Noir" (Dark mode by default, high contrast neon accents for different layers).
    - L1 Strategic: Purple (Deep thought).
    - L2 Tactical: Blue (Organization).
    - L3 Execution: Green (Action).
    - Healing/Errors: Orange/Red.
- **Typography:** Monospace for code/logs, Sans-serif (Inter) for UI controls.

## 5. Metric for Success: "Time to Trust"
We measure UX success by how many tasks the user allows the agent to run **without** manual approval over time.
- Day 1: 10% autonomous.
- Day 30: 80% autonomous.

## 6. Roadmap
1.  **Phase 1:** Implement `Rich` CLI enhancements (The "Pro" Look).
2.  **Phase 2:** Build the `EventBridge` to push data to the web.
3.  **Phase 3:** Launch the React-based `Holodeck` v1.0.
