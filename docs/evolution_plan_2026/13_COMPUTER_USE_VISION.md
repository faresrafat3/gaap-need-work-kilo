# GAAP Evolution: Universal Computer Use (Vision & Control)

**Focus:** Enabling GAAP to use the computer like a human (See Screen -> Move Mouse -> Click).

## 1. The Limitation of Text
Current GAAP is blind. It can only run CLI commands. It cannot:
- Use a web browser to fill a complex form.
- Watch a video to summarize it.
- Use GUI-only software (like Photoshop or obscure trading platforms).

**Target:** **Visual Action Model (VAM).**

## 2. Architecture: The "Vision-Action" Loop

We will implement a new layer: `Layer3_Visual`.

### 2.1 The Components
1.  **The Eye (Screen Capture):**
    - Uses `pyautogui` or `mss` to take screenshots of the desktop or specific windows.
    - Sends image + user prompt to a Vision Model (e.g., GPT-4o, Claude 3.5 Sonnet, or local LLaVA).
2.  **The Brain (Coordinate Inference):**
    - Prompt: "Where is the 'Export' button in this screenshot?"
    - Response: `{"element": "export_btn", "coordinates": [1024, 768]}`.
3.  **The Hand (HID Control):**
    - Uses `pyautogui` to move mouse to `[x, y]` and click.
    - Uses `keyboard` library to send keystrokes.

### 2.2 Safety Protocol (The "Leash")
Giving AI mouse control is dangerous.
- **Fail-safe:** Moving the mouse to the top-left corner (0,0) immediately kills the agent process.
- **Sandboxing:** Run the GUI apps inside a VNC / Docker container so the agent doesn't mess up the host OS.

## 3. Implementation Plan
1.  **Phase 1:** Build `ScreenReader` tool (OCR + Object Detection).
2.  **Phase 2:** Implement `MouseController` wrapper with safety limits.
3.  **Phase 3:** Create `VisualPlanner` in Layer 2 that breaks tasks into visual steps (e.g., "Open Browser", "Click Search", "Type Query").
