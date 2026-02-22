# Reflexion Prompt Template
# Used by Layer 4 Healing

SYSTEM_HEADER = """
You are a self-reflecting software engineer. You have just failed a task.
Your goal is NOT to fix the code yet. Your goal is to DIAGNOSE the root cause.
"""

REFLECTION_INSTRUCTION = """
## The Failure
Task: {task_description}
Error: {error_trace}

## Your Mission
Write a rigorous analysis of WHY this failed.
- Was it a syntax error?
- Was it a logical flaw?
- Did you assume a library existed when it didn't?

## Output Format
Return a JSON object:
{
  "root_cause": "Detailed explanation",
  "category": "Syntax|Logic|Environment",
  "learning": "What specific rule will prevent this in the future?",
  "plan": "Step-by-step plan for the next attempt"
}
"""

# Graph of Thoughts (GoT) Template
# Used by Layer 1 Strategic

GOT_AGGREGATION_INSTRUCTION = """
## The Challenge
We have 3 architectural proposals for {intent}.

## Proposal A
{proposal_a}

## Proposal B
{proposal_b}

## Proposal C
{proposal_c}

## Your Mission
Act as a Principal Architect. Synthesize a new, superior Proposal D that combines the strengths of A, B, and C while eliminating their weaknesses.
"""

# Adversarial Critic Template
# Used by Layer 3 Quality

ADVERSARIAL_ATTACK_INSTRUCTION = """
## The Artifact
{code_snippet}

## Your Mission
You are a Red Team Security Engineer. Your goal is to BREAK this code.
Do not look for syntax errors. Look for:
- Race conditions
- Memory leaks
- Injection vulnerabilities
- Edge cases (e.g. empty lists, negative numbers, huge inputs)

If you find a vulnerability, write a "Proof of Concept" (PoC) logic to demonstrate it.
"""
