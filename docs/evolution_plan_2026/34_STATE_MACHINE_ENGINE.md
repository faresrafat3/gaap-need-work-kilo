# TECHNICAL SPECIFICATION: State-Machine FSM Engine (v1.0)

**Author:** Strategic Architect
**Target:** Code Agent
**Focus:** Modeling the web application as a Finite State Machine to find transition vulnerabilities.

## 1. Concept: The Interaction Graph
Web apps are not static; they are dynamic states triggered by user actions (Clicks, Submits). Vulnerabilities often live in unauthorized transitions between these states.

## 2. Component Design

### 2.1 State Discovery (The Crawler)
- **Engine:** Playwright / Selenium (Headless).
- **Action:** Record the DOM state before and after every interaction.
- **State Identifier:** Generate a hash of the DOM structure (stripping dynamic content).
- **Result:** A Directed Graph where Nodes = Pages/States, Edges = Actions.

### 2.2 Transition Audit
Once the graph is built, GAAP attempts to:
- **Skip Transition:** Can I reach `State: OrderConfirmed` directly from `State: Cart` without passing through `State: Payment`?
- **Unauthorized Replay:** Can I replay the `Edge: SubmitPayment` multiple times (Race Condition)?
- **Privilege Jump:** If `Account A` can reach `State: AdminPanel`, can `Account B` (Low Privilege) reach it by replaying the same URL/Headers?

### 2.3 Differential Analysis (The Fractal Sync)
- Launch two Fractals: `Agent_Alpha` (High Perms) and `Agent_Beta` (Low Perms).
- Compare the Graph of Alpha vs Beta.
- Identify nodes present in Alpha but accessible in Beta via manual URL manipulation.

## 3. Implementation Steps
1. **Model** the FSM using `networkx`.
2. **Execute** browsing sessions using Playwright.
3. **Compare** response bodies using **Jaccard Similarity**; if similarity > 0.9 on an unauthorized request, it's a potential leak.
