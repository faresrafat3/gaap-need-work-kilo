# System Patterns: GAAP

## Architecture: Layered Hierarchy
- **Layer 0 (Interface):** Handles interaction with the user/external systems.
- **Layer 1 (Strategic):** High-level planning and goal setting.
- **Layer 2 (Tactical):** Breaking down plans into actionable tasks.
- **Layer 3 (Execution):** Direct interaction with tools and code.

## Key Patterns
- **Provider Abstraction:** Unified interface for multiple LLM backends.
- **Self-Healing Loop:** Monitor -> Detect Failure -> Generate Fix -> Apply -> Verify.
- **Memory Guard:** Ensures safe and efficient context window management.
- **Meta-Learning Loop:** Analyze session -> Extract patterns -> Update knowledge base.
