"""
GAAP MAD Module (Multi-Agent Debate)
=====================================

Implements consensus-building through critic panels:

Critic Types:
    - ArchitectureCriticType: Code architecture reviewers
    - SYSTEM_PROMPTS: Critic persona definitions
    - CRITIC_DESCRIPTIONS: Critic capabilities

Response Parsing:
    - parse_critic_response: Parse MAD evaluation results
    - parse_architecture_critic_response: Architecture-specific parsing
    - fallback_evaluation: Default evaluation when parsing fails

Features:
    - Multiple critic perspectives
    - Structured evaluation output
    - Quality scoring and consensus
    - Language detection

Usage:
    from gaap.mad import SYSTEM_PROMPTS, parse_critic_response

    prompt = build_user_prompt(artifact, task)
    response = await provider.chat(prompt)
    evaluation = parse_critic_response(response)
"""

from gaap.mad.critic_prompts import (
    ARCH_SYSTEM_PROMPTS,
    CRITIC_DESCRIPTIONS,
    SYSTEM_PROMPTS,
    ArchitectureCriticType,
    build_architecture_prompt,
    build_user_prompt,
    get_language_from_artifact,
)
from gaap.mad.response_parser import (
    CriticParseError,
    fallback_architecture_evaluation,
    fallback_evaluation,
    parse_architecture_critic_response,
    parse_critic_response,
)

__all__ = [
    "SYSTEM_PROMPTS",
    "ARCH_SYSTEM_PROMPTS",
    "ArchitectureCriticType",
    "build_user_prompt",
    "build_architecture_prompt",
    "get_language_from_artifact",
    "CRITIC_DESCRIPTIONS",
    "parse_critic_response",
    "parse_architecture_critic_response",
    "CriticParseError",
    "fallback_evaluation",
    "fallback_architecture_evaluation",
]
