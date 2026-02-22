# Changelog

All notable changes to GAAP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0-SOVEREIGN] - 2026-02-20

### "The Cognitive Awakening: Multi-Domain Intelligence"

#### Major Breakthroughs
- **Dynamic Cognitive Architecture**: Transformed L1, L2, and L3 from rigid software-focused pipelines into adaptive cognitive engines capable of handling Research, Diagnostics, and Coding with specialized strategies.
- **Sovereign Dashboard**: Introduced `OODA_Monitor.py`, a real-time war room visualizing the full Observe-Orient-Decide-Act loop.

#### Added
- **Layer 1 (Strategic)**:
    - Dynamic Strategy Selection: Auto-switches between `ARCH`, `RESEARCH`, and `DEBUG` thinking models.
    - Adaptive Tree of Thoughts: Generates task-specific thinking streams (e.g., Methodology -> Sources vs. Paradigm -> Database).
- **Layer 2 (Tactical)**:
    - Expanded Task Ontology: Added 15+ new task categories including `INFORMATION_GATHERING`, `ROOT_CAUSE_ANALYSIS`, and `DATA_SYNTHESIS`.
    - Context-Aware Decomposition: Decomposer now injects specific tech stacks and components into task descriptions.
- **Layer 3 (Execution)**:
    - **Persona Injection**: Agents dynamically adopt "Researcher", "Diagnostic Engineer", or "Software Engineer" personas.
    - **Tool Arsenal Filtering**: Security enforcement that restricts tools based on task category (e.g., Researchers cannot execute arbitrary code).
    - **Specialized MAD Panel**: Added `ACCURACY`, `SOURCE_CREDIBILITY`, and `ROOT_CAUSE` critics for non-code tasks.
- **Memory System**:
    - **Domain-Specific Lessons**: Episodic memory now categorizes lessons by domain, ensuring research tasks only recall research lessons.

#### Changed
- **Strategy Engine**: Replaced static `STRATEGY_PROMPT` with a dynamic prompt selector.
- **Task Decomposer**: Replaced static `DECOMPOSITION_PROMPT` with domain-specific templates.
- **Quality Assurance**: Overhauled `MADQualityPanel` to use dynamic weights and critic selection.

---

## [2.0.0-EVOLUTION] - 2026-02-18

### "The Great Awakening: Sovereign AGI Transition"

#### Added
- **Foundational Specs**: 50 High-Fidelity Technical Specifications (`docs/evolution_plan_2026/`) covering everything from Fractal Security to Recursive Self-Improvement.
- **Engine V2**: Fully implemented OODA Loop (Observe-Orient-Decide-Act) replacing the legacy linear pipeline.
- **Cognitive Modules**:
    - `gaap/research/`: Deep Research Engine using STORM (Perspective Synthesis) architecture.
    - `gaap/simulation/`: The Holodeck (GhostFS) for predictive execution and risk analysis.
    - `gaap/tools/`: Just-in-Time Tool Synthesizer for autonomous capability expansion.
    - `gaap/verification/`: Formal Verification using Z3 SMT Solver for mathematical logic proofs.
- **Swarm & Governance**:
    - `gaap/swarm/`: GISP Protocol (GAAP Internal Swarm) with Consensus Oracle and Arbitrator.
    - `gaap/meta_learning/`: Architecture Auditor for self-consistency and Prompt Breeder for Darwinian instruction evolution.
- **Security & Stealth**:
    - `gaap/security/`: DLP (Data Loss Prevention) Scanner with Entropy-based secret detection.
    - `gaap/web_recon/`: Stealth Agent with human-pacing and WAF evasion.
- **Memory & Identity**:
    - `gaap/memory/`: VectorStore implementation using ChromaDB for long-term semantic retrieval.
    - `.gaap/identity.json`: The system's "Soul" record (Version, Activated Features, Trust Scores).

#### Changed
- Architecture: Migrated from **Linear Waterfall** to **Circular OODA Loop**.
- Strategy: Upgraded from **Tree of Thoughts** to **Graph of Thoughts (GoT)**.
- Planning: Implemented **MCTS (Monte Carlo Tree Search)** for high-complexity decision making.
- Versioning: Jumped to `2.0.0-EVOLUTION` to reflect the sovereign AGI status.

---

## [1.0.0] - 2026-02-16

### Added

#### Core System
- 4-layer cognitive architecture (L0-L3)
- Comprehensive type system with 18 enums and 25+ dataclasses
- Hierarchical exception system with error codes and recovery suggestions
- Thread-safe configuration manager with hot reload support
- Fluent configuration builder pattern

#### Layers
- **Layer 0 (Interface)**: Security scanning, intent classification (11 types), complexity estimation, routing decisions
- **Layer 1 (Strategic)**: Tree of Thoughts exploration, MAD Architecture Panel, LLM-powered architecture generation
- **Layer 2 (Tactical)**: Task decomposition with DAG construction, cycle detection, critical path analysis
- **Layer 3 (Execution)**: Parallel execution, Genetic Twin verification, MAD Quality Panel with 6 critic types

#### Providers
- Groq provider (fastest - 227ms avg)
- Cerebras provider (reliable - 511ms avg)
- Gemini provider with key pool rotation
- Mistral provider
- GitHub Models provider
- G4F multi-provider support (free access to Gemini 2.5, GPT-4o-mini)
- WebChat providers (Kimi, DeepSeek, GLM)
- Unified provider with automatic failover
- Smart router with multi-strategy support

#### Self-Healing
- 5-level healing hierarchy (Retry -> Refine -> Pivot -> Strategy -> Human)
- Error classification system (Transient, Syntax, Logic, Model Limit, Resource, Critical)
- Prompt refinement templates
- Task simplification for complex failures

#### Memory
- 4-tier hierarchical memory (Working, Episodic, Semantic, Procedural)
- Memory decay calculation
- Persistence support (JSON)
- Pattern extraction from episodes

#### Security
- 7-layer prompt firewall
- Attack type detection (Injection, Jailbreak, Data Exfiltration, Code Injection)
- Audit trail with hash chain integrity
- Capability token system
- Contextual verification

#### CLI & Web
- Full-featured CLI with 8 command groups
- Streamlit web dashboard with 6 pages
- FastAPI REST API with OpenAPI docs

#### Observability
- OpenTelemetry tracing support
- Prometheus metrics endpoint
- Structured logging with structlog
- Rate limiting with 3 strategies (Token Bucket, Sliding Window, Adaptive)

### Changed
- Reduced healing retries (3,2,2,1 -> 1,1,1,1) to prevent timeout cascades
- Improved firewall critical pattern escalation
- Optimized memory guard with RSS monitoring

### Fixed
- Timeout handling in provider fallback
- Cycle detection in task graph
- Memory leak in long-running sessions

---

## Future Roadmap

### [1.1.0] - Planned Q2 2026
- Semantic intent classification with embeddings
- Embedding-based memory retrieval
- ML-based firewall detection
- Adaptive routing learning

### [1.2.0] - Planned Q3 2026
- Parallel layer execution
- Cost optimization routing
- Streaming support
- Multi-tenant support

### [2.0.0] - Planned Q4 2026
- RBAC system
- Enhanced audit logging
- Compliance support (SOC2, GDPR)
- Self-hosted deployment options

---

## [2.2.0] - 2026-02-22

### "Code Quality Renaissance"

#### Code Quality Improvements
- **Type Hints**: Increased coverage from 35% to 80% across the codebase
- **Docstrings**: Increased coverage from 15% to 65%
- **Deep Nesting**: Reduced from 13 levels to ~5 levels in layer3_execution.py
- **Silent Exceptions**: Fixed 27 instances of `except Exception: pass`
- **Type Ignores**: Reduced from 32 to 5 (84% improvement)

#### Architecture
- **WebChat Split**: Refactored webchat_providers.py (2057 lines → 8 modular files)
  - `gaap/providers/webchat/__init__.py` - Module exports
  - `gaap/providers/webchat/auth.py` - Authentication
  - `gaap/providers/webchat/base.py` - Base class
  - `gaap/providers/webchat/glm.py` - GLM provider
  - `gaap/providers/webchat/kimi.py` - Kimi provider
  - `gaap/providers/webchat/deepseek.py` - DeepSeek provider
  - `gaap/providers/webchat/copilot.py` - Copilot provider
  - `gaap/providers/webchat/registry.py` - Registry and CLI

#### New Modules
- **SOP Governance** (`gaap/core/governance.py`): Process-driven intelligence with role definitions, SOP steps, and mandatory artifact validation
- **Axiom Validator** (`gaap/core/axioms.py`): Constitutional gatekeeper with invariant checks (syntax, dependency, interface, read-only)
- **Observer** (`gaap/core/observer.py`): Environment scanning for OODA loop
- **Kilo Gateway** (`gaap/providers/kilo_gateway.py`): Access to 500+ models through unified endpoint

#### Removed (Dead Code)
- `gaap/context/` - External brain, HCL, orchestrator (~3000 lines)
- `gaap/meta_learning/` - Old meta-learning system (~1300 lines)
- `gaap/providers/unified_provider.py` - Old unified provider (1005 lines)
- `gaap/providers/free/` and `gaap/providers/free_tier/` - Old free providers (~1300 lines)
- `gaap/web/` - Old web interface (~1000 lines)
- `gaap/api/` - Old FastAPI app (~700 lines)
- `gaap/cache/` - Old caching system (~500 lines)
- `gaap/storage/` - Old storage layer (~300 lines)
- `gaap/validators/` - Old validators (~700 lines)

#### Module Coverage
| Module | Docstrings | Type Hints |
|--------|------------|------------|
| Core | 100% | 100% |
| Layers L0 | 100% | 100% |
| Memory | 95% | 95% |
| Providers | 85% | 85% |
| Overall | 65% | 80% |

#### Statistics
- **Python Files**: ~90 (down from 137)
- **Lines of Code**: ~35,000 (net reduction of ~8,500 lines)
- **Tests**: 49 passed, 2 skipped

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 2.2.0 | 2026-02-22 | **Code Quality Renaissance** (Type hints 80%, Docstrings 65%, Dead code removal) |
| 2.1.0 | 2026-02-20 | **Sovereign Cognitive Architecture** (Multi-Domain Intelligence) |
| 2.0.0 | 2026-02-18 | **Sovereign Transition** (OODA, GoT, MCTS) |
| 1.0.0 | 2026-02-16 | Initial release with 4-layer architecture |
