# ── FILE: cobol_moderniser/utils/graph_utils.py ───────────────────
"""
utils/graph_utils.py -- NetworkX helpers for COBOL dependency analysis.


Used by:
  agents/comprehension_agent.py  -- build_dependency_graph()
  agents/planning_agent.py       -- leaf detection, cycle detection
"""


from __future__ import annotations
import logging
from typing import Any
import networkx as nx


logger = logging.getLogger(__name__)




def build_dependency_graph(programs: list) -> nx.DiGraph:
    """
    Build a directed graph from a list of COBOLProgram objects.


    Nodes: program_id strings
    Edges: PERFORM (intra-program paragraph calls) and CALL (inter-program)
    Node attributes: filepath, paragraph_count, data_item_count, fan_in, fan_out
    """
    G = nx.DiGraph()
    program_index = {p.program_id: p for p in programs}


    # Add all known programs as nodes first
    for prog in programs:
        G.add_node(
            prog.program_id,
            filepath=prog.filepath,
            paragraph_count=len(prog.paragraphs),
            data_item_count=len(prog.all_data_items),
        )


    # Add edges from PERFORM and CALL statements
    for prog in programs:
        for para in prog.paragraphs:
            # Internal PERFORM (intra-program paragraph calls)
            for target in para.performs:
                if target in prog.paragraph_names:
                    # Self-loop with metadata (intra-program)
                    if not G.has_edge(prog.program_id, prog.program_id):
                        G.add_edge(prog.program_id, prog.program_id,
                                   type='PERFORM_INTERNAL', count=1)
                    else:
                        G[prog.program_id][prog.program_id]['count'] += 1


            # External CALL (inter-program)
            for target in para.calls:
                if target not in G:
                    # External dependency -- add placeholder node
                    G.add_node(target, filepath='external',
                               paragraph_count=0, data_item_count=0)
                G.add_edge(prog.program_id, target,
                           type='CALL', from_para=para.name)


    # Compute and store fan-in / fan-out on each node
    for node in list(G.nodes()):
        # Exclude self-loops when computing fan-in
        in_edges  = [u for u, v in G.in_edges(node)  if u != node]
        out_edges = [v for u, v in G.out_edges(node) if v != node]
        G.nodes[node]['fan_in']  = len(in_edges)
        G.nodes[node]['fan_out'] = len(out_edges)


    logger.info('Dependency graph: %d nodes, %d edges',
                G.number_of_nodes(), G.number_of_edges())
    return G




def get_leaf_programs(G: nx.DiGraph) -> list[str]:
    """
    Return program IDs with no outgoing inter-program edges.
    Leaf programs are the safest to migrate first (no downstream dependents).
    """
    leaves = []
    for node in G.nodes():
        if G.nodes[node].get('filepath') == 'external':
            continue
        out_real = [v for _, v in G.out_edges(node) if v != node]
        if not out_real:
            leaves.append(node)
    return leaves




def detect_cycles(G: nx.DiGraph) -> list[list[str]]:
    """Return list of cycles in the dependency graph (each cycle is a list of node IDs)."""
    try:
        return list(nx.simple_cycles(G))
    except Exception:
        return []




def topological_order(G: nx.DiGraph) -> list[str]:
    """
    Return programs in topological order (dependencies first).
    Falls back to degree-sorted order if cycles exist.
    """
    real_nodes = [n for n in G.nodes() if G.nodes[n].get('filepath') != 'external']
    sub = G.subgraph(real_nodes).copy()
    sub.remove_edges_from(nx.selfloop_edges(sub))
    try:
        return list(reversed(list(nx.topological_sort(sub))))
    except nx.NetworkXUnfeasible:
        logger.warning('Cycles detected -- using degree sort instead')
        return sorted(real_nodes, key=lambda n: G.nodes[n].get('fan_out', 0))




def graph_summary(G: nx.DiGraph) -> str:
    """Human-readable one-line summary of the dependency graph."""
    real = [n for n in G.nodes() if G.nodes[n].get('filepath') != 'external']
    ext  = [n for n in G.nodes() if G.nodes[n].get('filepath') == 'external']
    leaves = get_leaf_programs(G)
    cycles = detect_cycles(G)
    return (f'Graph: {len(real)} programs, {len(ext)} external deps, '
            f'{G.number_of_edges()} edges, '
            f'{len(leaves)} leaf nodes, {len(cycles)} cycles')
