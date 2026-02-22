# TECHNICAL SPECIFICATION: The Prompt Breeding Ground (APO Engine) - v2.0 (Deep Simulation Verified)

**Focus:** Transforming GAAP into a system that evolves its own instructions using Evolutionary Algorithms (Inspired by PromptBreeder & DSPy).

## 1. The Core Philosophy: Prompts as DNA
Static prompts are dead code. In GAAP, prompts are **living organisms** that evolve based on survival of the fittest.

## 2. Architecture: The Evolutionary Loop

We introduce a semi-detached module: **The Prompt Factory**.

### 2.1 The Gene Pool Taxonomy (`gaap/.kilocode/memory/prompts/`)
Prompts are categorized by **Species** to prevent harmful crossovers.
- `coding/`: Optimized for syntax correctness.
- `reasoning/`: Optimized for logic and CoT.
- `security/`: Optimized for paranoia and checking.
- `creative/`: Optimized for fluency and temperature.

**Genome Structure:**
```json
{
  "id": "gene_code_v12",
  "species": "coding",
  "instruction": "You are a Senior Python Engineer. Use Type Hints.",
  "examples": ["example_hash_1", "example_hash_2"],
  "fitness_history": [0.85, 0.92, 0.88],
  "parent_ids": ["gene_code_v9", "gene_style_v3"]
}
```

### 2.2 The Mutation Operator (`PromptMutator`)
An LLM Agent that acts as the "Biological Mutator".
- **Zero-Order (Rephrase):** "Write code" -> "Generate implementation".
- **First-Order (Add Constraint):** + "Do not use deprecated libraries".
- **Crossover (Hybridize):** Takes the *Persona* from Parent A and the *Constraints* from Parent B.

### 2.3 The Fitness Function (The Judge)
To avoid overfitting, fitness is calculated over a **Batch of 5 Diverse Tasks**.
`Fitness = (Avg(Hard_Metrics) * 0.7) + (Avg(Soft_Metrics) * 0.3) - (Token_Cost_Penalty)`

### 2.4 Drift Protection (The DNA Police)
Before accepting a child prompt, we run a **Semantic Similarity Check**.
- If `CosineSimilarity(Child, Original_Intent) < 0.85`: ABORT. (The prompt lost its meaning).

## 3. Workflow: "Breeding Season"
When the system is idle (Dreaming Cycle):
1.  **Select:** Pick top 2 Genomes from the `coding` pool.
2.  **Mutate & Cross:** Generate 10 candidates.
3.  **Simulate:** Run candidates against the **"Golden Test Suite"** (from File 45).
4.  **Update:** If Child > Parent, archive Parent and promote Child to `ACTIVE`.

## 4. Integration with Runtime
The `GAAPEngine` requests prompts dynamically:
- `prompt = PromptFactory.get_best("coding", context_size="small")`

## 5. Implementation Roadmap
1.  **Phase 1:** Build the `GenePool` storage using JSON/SQLite.
2.  **Phase 2:** Implement `PromptMutator` with safeguards (Drift Protection).
3.  **Phase 3:** Connect the loop to `scripts/evaluate_agent.py` as the fitness ground.

---
**Handover Status:** Verified via Mental Simulation. Ready for Code Agent.
