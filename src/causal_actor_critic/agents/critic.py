"""Constraint-aware Critic — validates SQL with deconfounding metadata.

Unlike a standard Critic that only sees the Actor's output, this Critic
also receives constraint metadata (query complexity, token budget, attempt
count).  This implements the backdoor adjustment: the Critic conditions on
the environmental constraints, separating Actor strategy quality from
environmental limitation.
"""

from __future__ import annotations

import json
import re

from causal_inference.llm_client import LLMClient
from causal_actor_critic.state import SQLActorCriticState

CRITIC_SYSTEM_PROMPT = """\
You are a meticulous SQL reviewer with CAUSAL AWARENESS.

You validate SQL queries against the user's question, Data Dictionary,
and Domain Rules. You also receive CONSTRAINT METADATA that tells you
about the operating environment:

- query_complexity: 0-1 score of how analytically complex the query is
- constraint_class: UNCONSTRAINED | COMPLEXITY_CONSTRAINED | SIZE_CONSTRAINED | CONTEXT_EXHAUSTED
- attempt: which correction attempt this is
- max_attempts: total allowed attempts
- complexity_factors: detected analytical patterns (window functions, CTEs, etc.)

## How to Use Constraint Metadata (CRITICAL)

When constraint_class is COMPLEXITY_CONSTRAINED:
- Evaluate the Actor's STRATEGY quality, not just output completeness
- Accept reasonable simplifications for genuinely complex queries
- Focus feedback on fixable logical errors, not on missing sophistication

When constraint_class is CONTEXT_EXHAUSTED:
- Be lenient — this is the last chance
- Accept best-effort output that addresses the core question

When constraint_class is UNCONSTRAINED:
- Apply the full rubric strictly

## Validation Rubric

1. Schema Correctness — table/column names exist, types correct
2. Logical Correctness — answers the question, correct JOINs, GROUP BY, filters
3. Domain Rule Compliance — follows business rules
4. SQL Best Practices — CTEs over subqueries, correct window functions
5. Security — no DML/DDL, no injection patterns

## Verdict Categories

- **pass**: Correct and complete (or correctly adapted to constraints).
- **salvageable**: Has fixable issues — provide corrected SQL.
- **non_salvageable**: Fundamental problems — Actor must regenerate.

## Output Format

Return STRICT JSON:
{
    "verdict": "pass" | "salvageable" | "non_salvageable",
    "issues": [{"category": "...", "severity": "...", "description": "..."}],
    "feedback": "Summary for the Actor. Empty if pass.",
    "corrected_sql": "Fixed SQL if salvageable. Empty otherwise.",
    "constraint_acknowledged": true/false
}
"""


async def critic_validate_sql(
    state: SQLActorCriticState, llm: LLMClient
) -> dict:
    """Validate the Actor's SQL with constraint-aware evaluation."""
    constraint_meta = state.get("constraint_metadata", {})

    prompt_parts = [
        f"## User's Original Question\n{state['user_query']}",
        f"## Generated SQL to Validate\n```sql\n{state['generated_sql']}\n```",
        f"## Actor's Explanation\n{state.get('sql_explanation', 'No explanation provided.')}",
        f"## Constraint Metadata (IMPORTANT — adjust your evaluation accordingly)\n"
        f"```json\n{json.dumps(constraint_meta, indent=2)}\n```",
    ]

    dd = state.get("data_dictionary", "")
    if dd:
        prompt_parts.append(f"## Data Dictionary\n{dd}")

    rules = state.get("domain_rules", "")
    if rules:
        prompt_parts.append(f"## Domain Rules\n{rules}")

    prompt_parts.append(
        "Evaluate this SQL considering the constraint metadata. "
        "Return your assessment as a JSON object."
    )

    user_prompt = "\n\n".join(prompt_parts)

    content = await llm.generate(
        system_prompt=CRITIC_SYSTEM_PROMPT, user_prompt=user_prompt
    )

    result = _parse_critic_response(content)

    return {
        "critic_verdict": result.get("verdict", "non_salvageable"),
        "critic_issues": result.get("issues", []),
        "critic_feedback": result.get("feedback", ""),
        "corrected_sql": result.get("corrected_sql", ""),
    }


def _parse_critic_response(content: str) -> dict:
    text = content.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {
        "verdict": "non_salvageable",
        "issues": [{"category": "logic", "severity": "high",
                     "description": "Critic response was not valid JSON."}],
        "feedback": f"Critic output could not be parsed. Raw:\n{content[:500]}",
        "corrected_sql": "",
    }
