"""
Causal-Inference Enhanced Actor-Critic Pattern for SQL Generation.

A LangGraph workflow where an Actor generates SQL and a Critic validates it,
enhanced with causal inference to resolve confounding, prevent deadlocks,
and ensure convergence of the correction loop.
"""

from causal_actor_critic.graph import build_causal_actor_critic_workflow

__all__ = ["build_causal_actor_critic_workflow"]
