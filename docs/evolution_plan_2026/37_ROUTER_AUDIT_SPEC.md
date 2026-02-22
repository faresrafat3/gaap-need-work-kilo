# TECHNICAL SPECIFICATION: Router Evolution (Model Cascading & MoA)

**Target:** `gaap/routing/router.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Inaccurate Pricing:** Static cost variables lead to budget drift.
- **Unfair Bias:** Prefers "Famous" models over specialized performers (e.g. DeepSeek for code).
- **One-Shot Routing:** No strategy for escalating from cheap to expensive models dynamically.

## 2. Refactoring Requirements

### 2.1 Implementing Live Cost Tracking
Update `BaseProvider` to include a `get_pricing()` method.
- **Requirement:** `estimated_cost = (input_tokens * p.price_in) + (output_tokens * p.price_out)`.

### 2.2 Task-Specific Performance Matrix
Replace static tier scores with a **Performance Heatmap**.
- **Data Source:** Fetch stats from `gaap/meta_learning/meta_learning.json`.
- **Logic:** If `DeepSeek` has a 95% success rate in `CODE_GEN` but 40% in `PLANNING`, adjust its score dynamically based on the current `TaskType`.

### 2.3 Model Cascading (The "Frugal" Strategy)
Implement a **Multi-Step Routing** logic.
1.  **Step 1:** Route to the cheapest model that meets the `min_quality`.
2.  **Step 2:** Validate output using `QualityGate`.
3.  **Step 3:** If quality < 0.7, **Escalate** to the next tier model using the same context.

### 2.4 Mixture-of-Agents (MoA) - Optional High ROI
For `CRITICAL` tasks, allow routing to **Two Providers simultaneously** and use a third cheap model to merge their answers.

## 3. Implementation Steps
1.  **Add** `pricing_table.json` to `gaap/core/`.
2.  **Update** `SmartRouter.route()` to support `step-wise` escalation.
3.  **Integrate** with `ExperienceAnalyzer` to feed real performance data back into routing scores.

---
**Handover Status:** Ready. Code Agent must implement Dynamic Escalation.
