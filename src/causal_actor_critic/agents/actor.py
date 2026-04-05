"""Actor node — generates SQL from natural language queries."""

from __future__ import annotations

import json
import re

from causal_inference.llm_client import LLMClient
from causal_actor_critic.state import SQLActorCriticState

ACTOR_SYSTEM_PROMPT = """\
You are an expert SQL analyst. Your task is to translate natural language
questions into precise, executable SQL queries.

## Core Responsibilities

1. Parse the user's question to identify needed data, filters, aggregation.
2. Generate a single SQL query using the Data Dictionary and Domain Rules.
3. Use CTEs over nested subqueries for readability.
4. Use window functions, self-joins, running totals when appropriate.
5. Explain your reasoning briefly.

## SQL Standards

- Write ANSI-compliant SQL.
- Only SELECT and WITH (CTE) statements. Never INSERT/UPDATE/DELETE/DROP/ALTER.
- Use LIMIT for "top N" requests.
- Alias columns with readable names using AS.

## When Receiving Feedback

If you receive feedback from a previous validation, address EVERY point.
Focus on the specific issues identified — do not restructure the entire
query unless the feedback explicitly says the approach is wrong.

## Output Format

### SQL
```sql
-- your query here
```

### Explanation
A concise paragraph explaining your approach.
"""


async def actor_generate_sql(
    state: SQLActorCriticState, llm: LLMClient
) -> dict:
    """Generate SQL from the user's natural language query."""
    attempt = state.get("attempt", 0) + 1

    parts = [f"## User Question\n{state['user_query']}"]

    dd = state.get("data_dictionary", "")
    if dd:
        parts.append(f"## Data Dictionary\n{dd}")

    rules = state.get("domain_rules", "")
    if rules:
        parts.append(f"## Domain Rules\n{rules}")

    feedback = state.get("critic_feedback", "")
    if feedback:
        parts.append(
            f"## Feedback from Previous Validation\n"
            f"Your previous SQL was rejected. Address every issue below:\n\n"
            f"{feedback}"
        )

    prev_sql = state.get("generated_sql", "")
    if prev_sql and feedback:
        parts.append(
            f"## Your Previous SQL (rejected)\n```sql\n{prev_sql}\n```"
        )

    user_prompt = "\n\n".join(parts)

    content = await llm.generate(
        system_prompt=ACTOR_SYSTEM_PROMPT, user_prompt=user_prompt
    )

    sql = _extract_sql(content)
    explanation = _extract_explanation(content)

    return {
        "generated_sql": sql,
        "sql_explanation": explanation,
        "attempt": attempt,
        "correction_history": [
            {"attempt": attempt, "source": "actor", "sql": sql}
        ],
    }


def _extract_sql(content: str) -> str:
    match = re.search(r"```sql\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_explanation(content: str) -> str:
    match = re.search(
        r"##\s*Explanation\s*\n(.*?)(?:\n##|\Z)", content, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    parts = re.split(r"```sql.*?```", content, flags=re.DOTALL | re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return ""
