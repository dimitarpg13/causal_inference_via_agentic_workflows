"""DAG visualization using networkx and matplotlib."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


def draw_dag(
    G: nx.DiGraph,
    *,
    title: str = "",
    node_color: str = "#74b9ff",
    latent_color: str = "#dfe6e9",
    edge_color: str = "#2d3436",
    font_size: int = 10,
    node_size: int = 2000,
    figsize: tuple[int, int] = (8, 5),
    layout: str = "dot",
    latent_nodes: set[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Render a networkx DiGraph as a causal DAG.

    Parameters
    ----------
    G : nx.DiGraph
        The causal graph.
    latent_nodes : set[str], optional
        Nodes to style as latent/unmeasured (dashed border, different colour).
    layout : str
        One of "dot", "spring", "shell", "kamada_kawai".
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    latent_nodes = latent_nodes or set()

    if layout == "dot":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except (ImportError, FileNotFoundError):
            pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    observed = [n for n in G.nodes if n not in latent_nodes]
    latent = [n for n in G.nodes if n in latent_nodes]

    nx.draw_networkx_nodes(
        G, pos, nodelist=observed, node_color=node_color,
        node_size=node_size, edgecolors=edge_color, linewidths=1.5, ax=ax,
    )
    if latent:
        nx.draw_networkx_nodes(
            G, pos, nodelist=latent, node_color=latent_color,
            node_size=node_size, edgecolors=edge_color, linewidths=1.5,
            node_shape="s", ax=ax,
        )

    nx.draw_networkx_edges(
        G, pos, edge_color=edge_color, arrows=True,
        arrowsize=20, arrowstyle="-|>", ax=ax,
        connectionstyle="arc3,rad=0.1",
    )

    nx.draw_networkx_labels(
        G, pos, font_size=font_size, font_weight="bold", ax=ax,
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_axis_off()

    if created_fig:
        fig.tight_layout()
    return fig if created_fig else None


def draw_dag_from_spec(
    dag_spec: dict[str, Any],
    **kwargs: Any,
) -> plt.Figure | None:
    """Build a networkx DiGraph from a DAG spec dict and render it.

    Parameters
    ----------
    dag_spec : dict
        Must have "nodes" and "edges" keys. May have "latent" (list of
        latent node names) and "description".
    """
    G = nx.DiGraph()
    G.add_nodes_from(dag_spec.get("nodes", []))
    G.add_edges_from(dag_spec.get("edges", []))

    latent = set(dag_spec.get("latent", []))
    title = kwargs.pop("title", dag_spec.get("description", ""))
    if len(title) > 80:
        title = title[:77] + "..."

    return draw_dag(G, title=title, latent_nodes=latent, **kwargs)
