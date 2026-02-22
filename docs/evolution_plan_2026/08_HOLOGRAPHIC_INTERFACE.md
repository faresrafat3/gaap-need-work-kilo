# GAAP Evolution: The Holographic Interface (UI/UX)

**Focus:** Making the invisible thought process visible and interactive.

## 1. The "Black Box" Problem
Currently, users see a spinner while GAAP "thinks". They have no idea if it's stuck, looping, or about to delete a database.
**Target:** A Real-Time "Brain Activity" Monitor.

## 2. Architecture: The React-Flow Dashboard

We will build a local web server (`gaap-ui`) separate from the core engine.

### 2.1 Tech Stack
- **Frontend:** Next.js + React Flow (for node graphs) + Framer Motion (animations).
- **Communication:** WebSockets (Socket.io) pushing state updates from `Layer3`.
- **Backend:** FastAPI (already existing in `gaap/api`).

### 2.2 Features

#### A. The "Thought Graph" Visualizer
Displays the **Tree of Thoughts (ToT)** as it grows.
- **Green Nodes:** Explored & Validated paths.
- **Red Nodes:** Dead ends / Rejected by Critic.
- **Pulsing Node:** Current active thought.
*Interaction:* User can click a "Green Node" to force the agent down that path, or click a "Red Node" to override the Critic.

#### B. The "Memory Heatmap"
Visualizes the **Vector Memory** cloud.
- Shows which "memories" are lighting up (being retrieved) based on the current context.
- Allows the user to inspect *why* a specific memory was recalled.

#### C. The "Live Code" Stream
A diff-view editor that shows code *as it is being written* character-by-character, allowing the user to hit a "PAUSE" button instantly.

## 3. User Experience (UX) Flow
1.  **User:** Types "Refactor the auth module" in CLI.
2.  **System:** Launches localhost:3000 automatically.
3.  **UI:** Shows the agent breaking down the task (Layer 2 DAG).
4.  **UI:** Shows the agent debating strategies (Layer 1 MAD).
5.  **User:** Sees the agent picking a risky strategy. User clicks "VETO" on the UI.
6.  **Agent:** "Strategy Vetoed. Switching to alternative B."

## 4. Roadmap
1.  **Phase 1:** Expose `EventStream` from `GAAPEngine`.
2.  **Phase 2:** Build basic React Flow prototype.
3.  **Phase 3:** Implement bi-directional control (Veto/Approve/Pause).
