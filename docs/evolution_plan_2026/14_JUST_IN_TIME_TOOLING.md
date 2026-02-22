# GAAP Evolution: Just-in-Time Tooling & Skill Synthesis

**Focus:** "Give a man a tool, he works for a day. Teach an agent to build tools, it works forever."

## 1. The Bottleneck of Hardcoded Tools
If we hardcode a `stock_trading_tool`, GAAP can trade stocks. If we hardcode a `video_editing_tool`, it can edit videos.
**Target:** The **Universal Tool Factory**.

## 2. Architecture: The Skill Synthesizer

When GAAP receives a request like "Analyze this audio file for sentiment", it checks its `ToolRegistry`. If no tool exists:

### 2.1 The "R&D Loop" (Research & Develop)
1.  **Research:** It searches the web/docs for "Python library for audio sentiment analysis".
    - Finds: `librosa`, `textblob`, `speech_recognition`.
2.  **Synthesis:** It writes a small Python script (`temp_tool_audio_sentiment.py`) that uses these libraries.
3.  **Validation:** It runs the script on a dummy file to ensure it works.
4.  **Register:** It adds this script to the `ToolRegistry` dynamically.
5.  **Execute:** It runs the user's request using the new tool.

### 2.2 The "Skill Cache" (Procedural Memory v2.0)
Instead of deleting the tool, it saves it to `gaap/skills/audio/`. Next time you ask, it uses the cached tool instantly.

## 3. Implementation Plan
1.  **Phase 1:** Build the `LibraryDiscoverer` agent (Search PyPI/GitHub).
2.  **Phase 2:** Build the `CodeSynthesizer` that writes standalone Python tools.
3.  **Phase 3:** Integrate with `Layer2_Tactical` to detect missing capabilities and trigger synthesis automatically.

## 4. Example Scenario: "Hack this Wi-Fi"
- User: "Audit my Wi-Fi security."
- GAAP: Checks tools -> None found.
- GAAP R&D: Searches "Python wifi audit". Finds `scapy`.
- GAAP Synthesis: Writes `wifi_scanner.py` using `scapy`.
- GAAP Execution: Runs the scanner, reports vulnerabilities.
