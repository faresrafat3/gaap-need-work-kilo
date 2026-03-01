# ADR-001: Why OODA Architecture

## Status
Accepted

## Context

GAAP needed a cognitive architecture that could:
- Make decisions under uncertainty
- Adapt quickly to changing requirements
- Learn from experience
- Handle complex, multi-step tasks

Traditional approaches like simple request-response or fixed pipelines didn't provide the adaptability needed for an autonomous AI agent.

## Decision

We adopted the OODA (Observe-Orient-Decide-Act) loop as the core cognitive architecture for GAAP.

## OODA Loop Overview

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Observe │────►│ Orient  │────►│ Decide  │────►│   Act   │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
      ▲                                               │
      └───────────────────────────────────────────────┘
                         (Feedback Loop)
```

## GAAP's 4-Layer Implementation

| Phase | Layer | Purpose | Key Components |
|-------|-------|---------|----------------|
| **Observe** | Layer 0 | Security, intent classification, routing | PromptFirewall, IntentClassifier, SmartRouter |
| **Orient** | Layer 1 | Strategic planning, research | ToTStrategic, MADPanel, MCTSStrategic |
| **Decide** | Layer 2 | Task decomposition, planning | TacticalDecomposer, TaskGraph, PhasePlanner |
| **Act** | Layer 3 | Execution with self-healing | ToolExecutor, SelfHealingSystem, CodeAuditor |

## Alternatives Considered

### REACT (Reasoning + Acting)
- **Pros:** Simple, well-known in LLM community
- **Cons:** Limited to single-turn reasoning, no strategic planning
- **Verdict:** Too simple for complex autonomous tasks

### Reflexion
- **Pros:** Self-reflection capabilities
- **Cons:** Lacks structured decision-making
- **Verdict:** Good for learning, insufficient for planning

### LangChain Agents
- **Pros:** Rich ecosystem, pre-built tools
- **Cons:** Black-box decision making, limited observability
- **Verdict:** Good for prototyping, insufficient for production autonomy

### Custom Pipeline
- **Pros:** Full control, simple
- **Cons:** Rigid, hard to adapt
- **Verdict:** Insufficient flexibility

## Why OODA Won

1. **Speed of Iteration**: "The entity that completes the OODA loop fastest wins" - John Boyd
2. **Observability**: Clear phase boundaries enable monitoring and debugging
3. **Adaptability**: Each phase can evolve independently
4. **Cognitive Science Foundation**: Based on proven military decision-making framework
5. **Layered Complexity**: Simple tasks skip layers, complex tasks use all

## Consequences

### Positive
- Clear mental model for developers
- Natural debugging through phase tracing
- Easy to explain to users (OODA visualizer)
- Each layer can be tested independently
- Scales from simple to complex tasks

### Negative
- Higher initial complexity than simple approaches
- More boilerplate code
- Learning curve for contributors

## Related Decisions

- [ADR-002: Database Choice](./002-database-choice.md)
- [ADR-003: Provider Abstraction](./003-provider-abstraction.md)

## References

- John Boyd, "The Essence of Winning and Losing"
- "Science, Strategy and War" by Frans Osinga
- https://www.oodaloop.com/
