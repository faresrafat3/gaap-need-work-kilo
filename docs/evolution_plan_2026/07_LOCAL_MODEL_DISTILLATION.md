# GAAP Evolution: Local Model Distillation & SLM

**Focus:** Reducing dependency on expensive, slow Cloud LLMs.

## 1. The Strategy: "Collect -> Distill -> Deploy"

AGI doesn't mean "The biggest model". It means "The most specialized intelligence for the task".

### 1.1 The Lifecycle
1.  **Collection:** GAAP uses Llama 3.3 (70B) or DeepSeek (67B) to solve complex architectural problems.
2.  **Distillation:** Successful logic and code generation patterns are extracted and formatted as `Instruction-Output` pairs.
3.  **Local Training:** A smaller model (e.g., Llama 3.1 8B or Mistral 7B) is fine-tuned (using LoRA/QLoRA) on this specialized dataset.
4.  **Specialization:** The result is a "GAAP-Specialized SLM" that runs locally and understands the specific coding style and project context perfectly.

## 2. Automated Dataset Generator (`DatasetForge`)

We will build a component that runs during the "Dreaming Cycle".

### 2.1 Forge Logic
- Filter `EpisodicMemory` for tasks with `quality_score > 90`.
- Extract the `StructuredIntent` (Input) and the `FinalArtifact` (Output).
- Clean the data (remove secrets, PII).
- Format into JSONL (OpenAI/HuggingFace format).

## 3. Local Model Router

Update `SmartRouter` to prefer the **Local SLM** for:
- Routine code generation (L3).
- Simple unit test writing.
- Documentation updates.
- Security scanning.

**Only escalate to Cloud Models for:**
- High-level Strategy (L1).
- Complex Architecture (L1).
- Critical failure recovery (Healing L4).

## 4. Hardware Requirements
- **Target:** 8GB-16GB VRAM (Consumer GPUs like RTX 3060/4060).
- **Inference Engine:** `llama.cpp` or `vLLM` or `Ollama`.

## 5. Roadmap
1.  **Phase 1:** Implement `DatasetForge` to start collecting training data today.
2.  **Phase 2:** Design the fine-tuning pipeline (scripts for Unsloth/Axolotl).
3.  **Phase 3:** Integrate the local inference engine as a "Provider" in `gaap/providers/`.
