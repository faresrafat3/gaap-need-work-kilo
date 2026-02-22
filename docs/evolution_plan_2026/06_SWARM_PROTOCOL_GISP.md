# GAAP Evolution: The Decentralized Swarm Marketplace (v2.0)

**Focus:** Moving from "Command & Control" to "Autonomous Negotiation".

## 1. The Core Philosophy: Intelligence as a Market
Static delegation is inefficient. In a complex project, different "Fractals" (Sub-agents) might have different success rates for specific tasks.
**Target:** **The Reputation-Based Task Auction (RBTA)**.

## 2. Architecture: The GISP v2.0 Protocol

Messages now include **Bid** and **Reputation** metadata.

### 2.1 The Negotiation Flow
1.  **Task Broadcast:** Orchestrator sends a `TASK_AUCTION` with requirements.
2.  **Bidding:** Fractals reply with a `TASK_BID` containing:
    - `estimated_success_rate`: Based on past Episodic Memory.
    - `estimated_cost`: Tokens/Time.
    - `reputation_score`: A metric of past performance in this specific domain (e.g., Python, SQL).
3.  **Award:** Orchestrator assigns the task to the fractal with the highest **Utility Score**.

### 2.2 The Reputation Engine
Every Fractal has a persistent profile.
- **Success +1:** Increases domain reputation.
- **Failure -1:** Decreases reputation, triggers a **Healing Audit**.
- **Metacognition:** If a Fractal correctly predicts its own failure (Epistemic Doubt), its reputation is *saved* from a penalty.

## 3. Communication Patterns (GISP v2.0)

#### A. TASK_BID (New)
```json
{
  "type": "TASK_BID",
  "task_id": "req_992",
  "bidder_id": "CoderFractal_01",
  "utility": 0.95,
  "rationale": "I have successfully implemented 5 similar Pydantic models in this project."
}
```

#### B. CONSENSUS_VOTE (New)
Used during Layer 1 Strategic debates. If 3 Fractals agree on a path, it is committed to memory as a **"Standard Operating Procedure (SOP)"**.

## 4. Emergent Behavior: "The Swarm Mind"
As reputation grows, Fractals will naturally form **"Guilds"**.
- A Guild for Frontend.
- A Guild for Security.
They will exchange "Memories" directly without the Orchestrator, leading to faster execution.

## 5. Implementation Roadmap
1.  **Phase 1:** Implement the `ReputationStore` in `gaap/memory/`.
2.  **Phase 2:** Update `GISP` schemas to support bidding and utility scores.
3.  **Phase 3:** Pilot the "Auction System" with 3 specialized Fractals.

