# GAAP Evolution: Multi-Modal IO Bus

**Focus:** Moving from Text to Everything.

## 1. The Bottleneck of Text-Only
Current GAAP can't see the chart on your screen or hear a bug in the audio. It relies on logs.
**Target:** The **Universal IO Bus**.

## 2. Architecture: The "Binary" Middleware

We will upgrade `Layer0_Interface` to accept multi-modal data.

### 2.1 The Inputs
- **Image:** Screenshots, diagrams, charts.
- **Audio:** Voice commands, error beeps, music analysis.
- **Video:** Live camera feed, YouTube links, tutorials.

### 2.2 The "Decoders" (Transcoders)
Every input passes through a "Decoder" before reaching `Layer1`:
1.  **Image -> VLM (GPT-4o/Claude) -> Description/Analysis.**
2.  **Audio -> Whisper/Faster-Whisper -> Transcription.**
3.  **Video -> Frame Sampler -> Scene Analysis.**

## 3. The "Generative Output"
GAAP can't just speak. It needs to produce:
1.  **Image:** `Stable Diffusion` pipeline integrated.
2.  **Audio:** `Bark` / `XTTS` for voice responses.
3.  **Video:** `ffmpeg` command generator for simple edits.

### 4. Implementation Plan
1.  **Phase 1:** Add `Image` support to `Layer0` (Base64/URL).
2.  **Phase 2:** Integrate `Whisper` for audio-in.
3.  **Phase 3:** Integrate `Stable Diffusion` for image-out.
