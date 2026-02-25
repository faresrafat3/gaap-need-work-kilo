"""
GAAP Swarm Intelligence System (GISP v2.0)

A decentralized marketplace for autonomous agent collaboration.

Key Concepts:
- **Reputation-Based Task Auction**: Tasks are auctioned to fractals based on
  their historical performance, not static assignment.
- **Emergent Guilds**: Fractals naturally form guilds based on domain expertise.
- **Epistemic Humility**: Fractals can predict their own failures to avoid
  reputation penalties.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SwarmOrchestrator                        │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │                  TaskAuctioneer                      │   │
    │  │    ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
    │  │    │ Bid #1  │  │ Bid #2  │  │ Bid #3  │            │   │
    │  │    │ U=0.94  │  │ U=0.87  │  │ U=0.72  │            │   │
    │  │    └─────────┘  └─────────┘  └─────────┘            │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                           │                                │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │                   ReputationStore                     │  │
    │  │   Fractal     │ Python │ SQL │ Security │ Overall   │  │
    │  │   Coder_01    │  0.94  │ 0.82│   0.71   │   0.85    │  │
    │  │   Analyst_02  │  0.67  │ 0.91│   0.45   │   0.68    │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                           │                                │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │                      Guilds                           │  │
    │  │   [Python Guild]  [SQL Guild]  [Security Guild]       │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from gaap.swarm import SwarmOrchestrator, FractalAgent, ReputationStore

    # Setup
    reputation = ReputationStore()
    orchestrator = SwarmOrchestrator(reputation_store=reputation)

    # Register fractals
    orchestrator.register_fractal(FractalAgent(
        fractal_id="coder_01",
        specialization="python",
        provider=my_provider,
    ))

    # Process task through auction
    result = await orchestrator.process_task(task)
"""

from gaap.swarm.reputation import (
    ReputationStore,
    ReputationEntry,
    ReputationScore,
    DomainExpertise,
)
from gaap.swarm.gisp_protocol import (
    GISPMessage,
    TaskAuction,
    TaskBid,
    TaskAward,
    ConsensusVote,
    MessageType,
)
from gaap.swarm.auction import (
    TaskAuctioneer,
    AuctionResult,
    UtilityScore,
)
from gaap.swarm.fractal import (
    FractalAgent,
    FractalState,
    FractalCapability,
)
from gaap.swarm.guild import (
    Guild,
    GuildMembership,
    GuildProposal,
)
from gaap.swarm.orchestrator import (
    SwarmOrchestrator,
    SwarmConfig,
    SwarmMetrics,
)

# Profile Evolver (MorphAgent-inspired)
from gaap.swarm.profile_evolver import (
    EvolutionStatus,
    EvolutionTrigger,
    PerformanceSnapshot,
    ProfileEvolution,
    ProfileEvolver,
    create_profile_evolver,
)

__all__ = [
    "ReputationStore",
    "ReputationEntry",
    "ReputationScore",
    "DomainExpertise",
    "GISPMessage",
    "TaskAuction",
    "TaskBid",
    "TaskAward",
    "ConsensusVote",
    "MessageType",
    "TaskAuctioneer",
    "AuctionResult",
    "UtilityScore",
    "FractalAgent",
    "FractalState",
    "FractalCapability",
    "Guild",
    "GuildMembership",
    "GuildProposal",
    "SwarmOrchestrator",
    "SwarmConfig",
    "SwarmMetrics",
    # Profile Evolver
    "EvolutionTrigger",
    "EvolutionStatus",
    "ProfileEvolution",
    "PerformanceSnapshot",
    "ProfileEvolver",
    "create_profile_evolver",
]
