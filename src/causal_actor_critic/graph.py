"""LangGraph graph for the causal-enhanced Actor-Critic SQL workflow.

Graph topology (causal-enhanced):

    START → assemble_context → generate_sql → collect_constraints
              → validate_sql → causal_diagnose → causal_route
                                                    │
                         ┌──────────────────────────┼──────────────────┐
                         ↓                          ↓                  ↓
                    reroute_actor           apply_correction    accept_with_caveat
                         │                          │                  │
                         ↓                          ↓                  ↓
                    generate_sql            collect_constraints      END
                                                    │
                                              validate_sql → ...

                    finalize → END

Key difference from standard Actor-Critic:
  - Constraint collector sits between Actor and Critic (deconfounding layer)
  - Causal diagnosis after Critic rejection determines root cause
  - Three-way routing: reroute_actor / apply_correction / accept_with_caveat
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from causal_inference.llm_client import LLMClient
from causal_actor_critic.agents.actor import actor_generate_sql
from causal_actor_critic.agents.critic import critic_validate_sql
from causal_actor_critic.agents.causal_router import (
    accept_with_caveat,
    apply_correction,
    causal_diagnose,
    causal_route,
    finalize,
)
from causal_actor_critic.constraint_collector import collect_constraints
from causal_actor_critic.state import SQLActorCriticState


_TPC_H_DATA_DICTIONARY = """\
## Tables

### nation
| Column | Type | Description |
|--------|------|-------------|
| n_nationkey | INTEGER | Primary key |
| n_name | CHAR(25) | Nation name |
| n_regionkey | INTEGER | FK → region.r_regionkey |
| n_comment | VARCHAR(152) | Comment |

### region
| Column | Type | Description |
|--------|------|-------------|
| r_regionkey | INTEGER | Primary key |
| r_name | CHAR(25) | Region name |
| r_comment | VARCHAR(152) | Comment |

### customer
| Column | Type | Description |
|--------|------|-------------|
| c_custkey | INTEGER | Primary key |
| c_name | VARCHAR(25) | Customer name |
| c_address | VARCHAR(40) | Address |
| c_nationkey | INTEGER | FK → nation.n_nationkey |
| c_phone | CHAR(15) | Phone number |
| c_acctbal | DECIMAL(15,2) | Account balance |
| c_mktsegment | CHAR(10) | Market segment |
| c_comment | VARCHAR(117) | Comment |

### orders
| Column | Type | Description |
|--------|------|-------------|
| o_orderkey | INTEGER | Primary key |
| o_custkey | INTEGER | FK → customer.c_custkey |
| o_orderstatus | CHAR(1) | 'F'=fulfilled, 'O'=open, 'P'=partial |
| o_totalprice | DECIMAL(15,2) | Total price |
| o_orderdate | DATE | Order date |
| o_orderpriority | CHAR(15) | Priority |
| o_clerk | CHAR(15) | Clerk |
| o_shippriority | INTEGER | Shipping priority |
| o_comment | VARCHAR(79) | Comment |

### lineitem
| Column | Type | Description |
|--------|------|-------------|
| l_orderkey | INTEGER | FK → orders.o_orderkey |
| l_partkey | INTEGER | FK → part.p_partkey |
| l_suppkey | INTEGER | FK → supplier.s_suppkey |
| l_linenumber | INTEGER | Line number |
| l_quantity | DECIMAL(15,2) | Quantity |
| l_extendedprice | DECIMAL(15,2) | Extended price |
| l_discount | DECIMAL(15,2) | Discount (0.00-1.00) |
| l_tax | DECIMAL(15,2) | Tax rate |
| l_returnflag | CHAR(1) | Return flag |
| l_linestatus | CHAR(1) | Line status |
| l_shipdate | DATE | Ship date |
| l_commitdate | DATE | Commit date |
| l_receiptdate | DATE | Receipt date |
| l_shipinstruct | CHAR(25) | Shipping instructions |
| l_shipmode | CHAR(10) | Ship mode |
| l_comment | VARCHAR(44) | Comment |

### supplier
| Column | Type | Description |
|--------|------|-------------|
| s_suppkey | INTEGER | Primary key |
| s_name | CHAR(25) | Supplier name |
| s_address | VARCHAR(40) | Address |
| s_nationkey | INTEGER | FK → nation.n_nationkey |
| s_phone | CHAR(15) | Phone |
| s_acctbal | DECIMAL(15,2) | Account balance |
| s_comment | VARCHAR(101) | Comment |

### part
| Column | Type | Description |
|--------|------|-------------|
| p_partkey | INTEGER | Primary key |
| p_name | VARCHAR(55) | Part name |
| p_mfgr | CHAR(25) | Manufacturer |
| p_brand | CHAR(10) | Brand |
| p_type | VARCHAR(25) | Type |
| p_size | INTEGER | Size |
| p_container | CHAR(10) | Container |
| p_retailprice | DECIMAL(15,2) | Retail price |
| p_comment | VARCHAR(23) | Comment |

### partsupp
| Column | Type | Description |
|--------|------|-------------|
| ps_partkey | INTEGER | FK → part.p_partkey |
| ps_suppkey | INTEGER | FK → supplier.s_suppkey |
| ps_availqty | INTEGER | Available quantity |
| ps_supplycost | DECIMAL(15,2) | Supply cost |
| ps_comment | VARCHAR(199) | Comment |
"""

_TPC_H_DOMAIN_RULES = """\
## Revenue Calculation
Revenue = l_extendedprice * (1 - l_discount). Always apply this formula.

## Date Ranges
Unless specified, use the full date range available in the data.

## Nation/Region Joins
Always join nation to region via n_regionkey = r_regionkey.
Always join customer to nation via c_nationkey = n_nationkey.
Always join supplier to nation via s_nationkey = n_nationkey.

## Order Status Codes
- 'F' = fulfilled, 'O' = open, 'P' = partially fulfilled

## Default Ordering
Unless the user specifies otherwise, order results by the primary metric descending.
"""


def build_causal_actor_critic_workflow(llm: LLMClient) -> Any:
    """Build and compile the causal-enhanced Actor-Critic SQL workflow."""

    async def _assemble_context(state: SQLActorCriticState) -> dict:
        return {
            "data_dictionary": state.get("data_dictionary") or _TPC_H_DATA_DICTIONARY,
            "domain_rules": state.get("domain_rules") or _TPC_H_DOMAIN_RULES,
            "max_attempts": state.get("max_attempts", 3),
            "attempt": 0,
            "correction_history": [],
        }

    async def _generate_sql(state: SQLActorCriticState) -> dict:
        return await actor_generate_sql(state, llm)

    async def _collect_constraints(state: SQLActorCriticState) -> dict:
        return collect_constraints(state)

    async def _validate_sql(state: SQLActorCriticState) -> dict:
        return await critic_validate_sql(state, llm)

    async def _causal_diagnose(state: SQLActorCriticState) -> dict:
        return await causal_diagnose(state, llm)

    builder = StateGraph(SQLActorCriticState)

    builder.add_node("assemble_context", _assemble_context)
    builder.add_node("generate_sql", _generate_sql)
    builder.add_node("collect_constraints", _collect_constraints)
    builder.add_node("validate_sql", _validate_sql)
    builder.add_node("causal_diagnose", _causal_diagnose)
    builder.add_node("apply_correction", apply_correction)
    builder.add_node("accept_with_caveat", accept_with_caveat)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "assemble_context")
    builder.add_edge("assemble_context", "generate_sql")
    builder.add_edge("generate_sql", "collect_constraints")
    builder.add_edge("collect_constraints", "validate_sql")
    builder.add_edge("validate_sql", "causal_diagnose")

    builder.add_conditional_edges(
        "causal_diagnose",
        causal_route,
        {
            "finalize": "finalize",
            "reroute_actor": "generate_sql",
            "apply_correction": "apply_correction",
            "accept_with_caveat": "accept_with_caveat",
        },
    )

    builder.add_edge("apply_correction", "collect_constraints")
    builder.add_edge("accept_with_caveat", END)
    builder.add_edge("finalize", END)

    return builder.compile()
