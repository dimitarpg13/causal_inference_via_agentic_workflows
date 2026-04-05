# Causal-Inference Enhanced Actor-Critic Design Pattern

## Overview

The standard **Actor-Critic** design pattern for LLM-based agentic systems pairs a Generator (Actor) that produces outputs with a Validator (Critic) that evaluates them. The adversarial tension between the two agents yields higher-quality results than either agent alone. However, the correction loop between Actor and Critic is susceptible to **confounding** — hidden variables that simultaneously affect the Actor's output quality and the Critic's ability to evaluate it, causing misattributed feedback signals, oscillation, and deadlocks.

This document describes how **causal inference** — specifically Pearl's causal hierarchy, Structural Causal Models (SCMs), and the backdoor adjustment — resolves these failure modes and makes the Actor-Critic loop provably convergent.

The running example is **SQL generation**: an Actor generates SQL from natural language, and a Critic validates it against a schema and domain rules. The pattern generalizes to any generation-validation workflow.

---

## 1. The Confounding Problem in Actor-Critic Systems

### Why Correction Loops Fail

In an Actor-Critic loop, the Critic observes the Actor's output and issues a verdict: pass, salvageable (fixable), or non-salvageable (regenerate). The verdict drives the Actor's next attempt. But when a **hidden variable** influences both the Actor's output quality and the Critic's evaluation, the feedback loop becomes confounded:

```
                 Confounder (U)
               (unobserved by Critic)
              /                      \
             ↓                        ↓
     Actor Output  ─────→  Critic Evaluation
             ↑                        │
             └── Correction Feedback ←─┘
```

The Critic cannot distinguish whether a bad outcome was caused by the **Actor's strategy** (fixable via correction) or by an **environmental variable** (not fixable by the Actor). When the Critic attributes an environmentally caused failure to the Actor, the correction prompt sends the Actor searching for a solution that does not exist — the beginning of oscillation.

### Three Flavors of Confounding

| Flavor | Confounder | Mechanism | Example |
|--------|-----------|-----------|---------|
| **Task-Difficulty** | Intrinsic complexity of the user's query | Complex queries simultaneously cause lower Actor output quality AND stricter Critic evaluation | A multi-table analytical SQL query with window functions fails not because the Actor chose a bad strategy, but because the task is inherently hard |
| **Physical-Constraint** | System guardrails (token limits, timeouts) | A physical constraint prevents complete output AND causes the Critic to flag incompleteness | The query requires a 20-CTE response but the output limit caps at 4,096 tokens |
| **Temporal-State** | Accumulated context window state | Correction attempts fill context, degrading both Actor generation and Critic evaluation quality | After 3 failed attempts, context is polluted with contradictory feedback |

### The Formal Problem

The standard best-response dynamic computes:

$$BR_{Actor}(s_{Critic}) = \arg\max_{s_{Actor}} E[u_{Actor} \mid s_{Actor}, s_{Critic}]$$

This conditions on the Critic's strategy but does **not** block the backdoor path through the confounder $U$. The conditional expectation conflates the causal effect of the Actor's strategy with the confounding bias:

$$E[u \mid s_{Actor}, s_{Critic}] = \underbrace{E[u \mid do(s_{Actor}), do(s_{Critic})]}_{\text{true causal effect}} + \underbrace{\text{bias}(U \to s_{Actor}, U \to u)}_{\text{confounding bias}}$$

---

## 2. The Causal Fix: Deconfounding via Backdoor Adjustment

### Making Hidden Constraints Observable

The core insight: route **constraint metadata** to the Critic as structured input, blocking the backdoor path. The Critic's evaluation becomes conditional on the constraint:

$$P(\text{FAIL} \mid \text{output}, \text{constraints}) \neq P(\text{FAIL} \mid \text{output})$$

| Constraint Class | Trigger | Critic Behavior |
|-----------------|---------|-----------------|
| `UNCONSTRAINED` | No guardrails active | Full rubric — hold Actor to complete standard |
| `COMPLEXITY_CONSTRAINED` | Query complexity score > threshold | Relax completeness, evaluate strategy quality |
| `SIZE_CONSTRAINED` | Output exceeds token limit | Accept partial output if prioritization is sound |
| `CONTEXT_EXHAUSTED` | Context utilization > 90% or final attempt | Accept best-effort; flag limitation for user |

### The Constraint Collection Layer

A **constraint collector** sits between the Actor's output and the Critic's evaluation, gathering active guardrail signals and attaching them as structured metadata:

```
  Actor Output ──→ Constraint Collector ──→ Critic Input (output + metadata)
                         ↑
            ┌────────────┼─────────────┐
            │            │             │
    Query Complexity   Token Budget   Attempt Counter
       Scorer          Monitor        / Context Monitor
```

### Causal Diagnosis: Strategy Failure vs Environmental Constraint

After a Critic rejection, a **Causal Diagnosis Agent** examines the constraint metadata and the Critic's issues to determine the root cause:

| Diagnosis | Meaning | Action |
|-----------|---------|--------|
| `ACTOR_STRATEGY_FAILURE` | The Actor chose a bad approach; the task is feasible | Re-route to Actor with targeted feedback |
| `ENVIRONMENTAL_CONSTRAINT` | A physical constraint prevented success; no Actor strategy can satisfy the Critic's rubric | Accept with caveat, or relax the Critic's rubric |
| `MIXED` | Some issues are Actor failures, others are constraint-driven | Fix what's fixable, caveat what isn't |

This diagnosis replaces the standard routing logic and prevents the deadlock cycle.

---

## 3. Enhanced Architecture

### Standard Actor-Critic (Confounded)

```
START → Actor → Critic → Router
                          │
              ┌───────────┼──────────┐
              ↓           ↓          ↓
            PASS    SALVAGEABLE   NON_SALVAGEABLE
              ↓           ↓          │
          Finalize   Apply Fix      │
                        ↓           │
                     Critic   ←─────┘
                                (re-generate)
```

### Causal-Enhanced Actor-Critic (Deconfounded)

```
START → Actor → Constraint Collector → Critic (constraint-aware)
                                         │
                                    Causal Diagnosis
                                         │
                    ┌────────────────────┼───────────────────────┐
                    ↓                    ↓                       ↓
              ACTOR_FAILURE        ENVIRONMENTAL            MIXED
                    ↓              CONSTRAINT                  ↓
            Re-route to Actor         ↓                 Fix Actor issues;
            with targeted        Accept with caveat     caveat the rest
            causal feedback      (relaxed rubric)
                    ↓                    ↓                       ↓
                 Critic           Synthesizer              Synthesizer
                    ↓                    ↓                       ↓
              (converges)           (terminates              (terminates
               or finalize          gracefully)              gracefully)
```

The key additions:
1. **Constraint Collector** between Actor and Critic
2. **Causal Diagnosis** after Critic rejection
3. **Three-way routing** based on causal attribution (not just verdict)
4. **Graceful termination** when environmental constraints make full success infeasible

---

## 4. Connection to Pearl's Causal Hierarchy

The three rungs of Pearl's causal ladder map directly onto the quality of Actor-Critic feedback:

| Rung | Pearl's Hierarchy | Actor-Critic Application | Outcome |
|------|------------------|-------------------------|---------|
| **L1: Association** | $P(Y \mid X)$ | Standard Critic: "output is incomplete" | Confounded — cannot distinguish Actor failure from constraint | 
| **L2: Intervention** | $P(Y \mid do(X))$ | Constraint-aware Critic: "output is incomplete AND constraint was active" | Deconfounded — correct causal attribution |
| **L3: Counterfactual** | $P(Y_x \mid X=x', Y=y')$ | "Had the Actor used a different strategy, would the output have passed?" | Targeted feedback — identifies which strategy dimension to change |

L2 is **necessary** for convergence. L3 **accelerates** convergence by providing more informative feedback.

### Counterfactual Credit Assignment

When the Actor changes multiple strategy dimensions between attempts (e.g., switches from nested subqueries to CTEs AND changes the join strategy), counterfactual reasoning isolates which change deserves credit:

$$\text{CF} = E[u(\text{CTE strategy}, \text{original joins}) \mid \text{observed}] - E[u(\text{original strategy}) \mid \text{observed}]$$

This tells the Actor: "the CTE switch was the real improvement; the join change was irrelevant." Without counterfactual credit, the Actor may lock onto an irrelevant change and miss the actual fix.

---

## 5. Convergence Conditions

A causal-enhanced Actor-Critic correction loop converges if and only if:

**Condition 1 — Feasibility:**
There exists at least one Actor strategy that the Critic will accept. When a guardrail makes the Critic's acceptance criterion physically unreachable, deconfounding restores feasibility by allowing the Critic to relax its rubric under active constraints.

**Condition 2 — Observability:**
Every variable that causally affects the Actor's output is observable by the Critic. The constraint collector ensures this by routing hidden constraint signals to the Critic as structured metadata.

**Condition 3 — Monotonicity:**
Each correction attempt moves the Actor closer to acceptance. Causal diagnosis ensures this by targeting feedback at the Actor's actual strategy failures rather than symptoms of hidden constraints.

---

## 6. SQL Generation: Concrete Application

### How the Causal Enhancement Works

1. **User submits query**: "Show monthly revenue trends by region with year-over-year growth rates"

2. **Constraint Collector scores complexity**: 
   - Multiple tables required: YES
   - Window functions needed: YES  
   - Temporal calculations: YES
   - Estimated complexity: HIGH (score 0.85)

3. **Actor generates SQL** (attempt 1): Produces a multi-CTE query with window functions

4. **Constraint Collector packages metadata**:
   ```json
   {
     "query_complexity": 0.85,
     "token_budget_remaining": 0.62,
     "attempt": 1,
     "max_attempts": 3,
     "context_utilization": 0.35,
     "constraint_class": "COMPLEXITY_CONSTRAINED"
   }
   ```

5. **Critic evaluates with constraint context**: Sees both the SQL AND the constraint metadata. Knows this is a complex query, so evaluates strategy quality rather than expecting perfection on first attempt.

6. **If rejected — Causal Diagnosis**:
   - Issue: "Missing GROUP BY column" → `ACTOR_STRATEGY_FAILURE` (fixable)
   - Issue: "Query too complex for single response" → `ENVIRONMENTAL_CONSTRAINT` (not fixable by Actor)

7. **Targeted feedback to Actor**: "Fix the GROUP BY clause. The query structure is sound for this complexity level."

### Without Causal Enhancement (Deadlock Scenario)

The same query without causal enhancement:
- Attempt 1: Actor generates complex SQL → Critic rejects: "Incomplete — missing YoY calculation"
- Attempt 2: Actor adds YoY → Critic rejects: "Logic error in window function frame"  
- Attempt 3: Actor restructures → Critic rejects: "Approach changed too much, lost original correctness"
- Result: `best_effort` with degraded quality

### With Causal Enhancement (Convergence)

- Attempt 1: Actor generates complex SQL → Constraint metadata shows HIGH complexity → Critic evaluates strategy, flags specific GROUP BY issue → Causal diagnosis: `ACTOR_STRATEGY_FAILURE` → targeted fix
- Attempt 2: Actor fixes GROUP BY → Critic passes with note about complexity-appropriate simplifications
- Result: `accepted` in 2 attempts

---

## 7. Implementation Notes

### Integration with Existing Causal Inference Workflow

This pattern builds on the causal inference package in `src/causal_inference/`:

- **State definitions** extend `CausalState` with Actor-Critic specific fields (constraint metadata, causal diagnosis)
- **LLM client** reuses the same `Settings` and `build_llm_client` infrastructure
- **LangGraph** orchestrates both the causal ladder agents and the Actor-Critic loop

### The Constraint Collector as Backdoor Adjustment

The constraint collector implements Pearl's backdoor adjustment in practice:

$$P(\text{FAIL} \mid \text{output}, \text{constraint}) = \text{(deconfounded Critic judgment)}$$

By conditioning on the constraint variable, we separate Actor strategy quality from environmental limitation — the same operation as conditioning on a confounder in a causal DAG.

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.
- Actor-Critic Agent Design Pattern — Documents 05 (Guardrail Design & Causal Analysis) and 08 (Causal Nash Equilibrium Convergence)
- Everett, P. & Fox, C. (2021). Causal Games: Unifying Strategic and Causal Reasoning.
- Yongacoglu, B. et al. (2024). Paths to Equilibrium in Games. NeurIPS 2024.
