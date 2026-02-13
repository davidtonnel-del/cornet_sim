#!/usr/bin/env python3
"""
CORNet Simulator – Closed Observational Relationalism Network analysis tool.
Performs bounded invariant checks, collapse detection, scale asymmetry diagnostics,
and density-dependent transition simulations.

Supports synthetic graphs and real X/Twitter thread ingestion.
"""
import argparse
import json
import random
import math
import numpy as np
import networkx as nx
from sympy import symbols, Eq, Function, Wild

# ────────────────────────────────────────────────
# Density estimation utilities
# ────────────────────────────────────────────────

def term_size(expr):
    if not expr.args:
        return 1
    return 1 + sum(term_size(a) for a in expr.args)

def random_term(variables, ops, max_size):
    if max_size <= 1 or random.random() < 0.3:
        return random.choice(variables)
    op = random.choice(ops)
    left_size = random.randint(1, max_size - 1)
    right_size = max_size - left_size
    return op(
        random_term(variables, ops, left_size),
        random_term(variables, ops, right_size),
    )

def wilson_interval(p_hat, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    radius = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2))) / denom
    return center - radius, center + radius

class EquationalPresentation:
    def __init__(self, generators, relations, associative_ops=None):
        self.generators = list(generators)
        self.relations = list(relations)
        self.associative_ops = set(associative_ops or [])

    def _right_associate(self, expr, op):
        if not expr.args or expr.func != op:
            return expr
        a, b = expr.args
        if a.func == op:
            x, y = a.args
            return self._right_associate(op(x, op(y, b)), op)
        return op(a, b)

    def normalize(self, expr, axioms=None, max_steps=15):
        if axioms is None:
            axioms = self.relations
        seen = set()
        cur = expr
        for _ in range(max_steps):
            if cur in seen:
                return cur
            seen.add(cur)
            new = cur
            for ax in axioms:
                new = new.replace(ax.lhs, ax.rhs)
                new = new.replace(ax.rhs, ax.lhs)
            for op in self.associative_ops:
                new = self._right_associate(new, op)
            if new == cur:
                break
            cur = new
        return cur

    def derive(self, eq, axioms=None, max_depth=5):
        if axioms is None:
            axioms = self.relations
        lhs = self.normalize(eq.lhs, axioms)
        rhs = self.normalize(eq.rhs, axioms)
        frontier = {lhs}
        seen = set(frontier)
        for _ in range(max_depth):
            next_frontier = set()
            for t in frontier:
                for ax in axioms:
                    for src, dst in ((ax.lhs, ax.rhs), (ax.rhs, ax.lhs)):
                        u = self.normalize(t.replace(src, dst), axioms)
                        if u == rhs:
                            return True
                        if u not in seen:
                            seen.add(u)
                            next_frontier.add(u)
            if not next_frontier:
                break
            frontier = next_frontier
        return False

def estimate_invariants_mc(pres, variables, k=6, samples=500, max_depth=3):
    ops = [g for g in pres.generators if callable(g)]
    hits = 0
    for _ in range(samples):
        t1 = pres.normalize(random_term(variables, ops, k))
        t2 = pres.normalize(random_term(variables, ops, k))
        if t1 != t2 and pres.derive(Eq(t1, t2), max_depth=max_depth):
            hits += 1
    p_hat = hits / samples
    return p_hat, wilson_interval(p_hat, samples)

# ────────────────────────────────────────────────
# Graph generation and loading
# ────────────────────────────────────────────────

def generate_synthetic_graph(n, p, seed, prefer_attach=False):
    random.seed(seed)
    G = nx.DiGraph()
    if prefer_attach:
        G = nx.barabasi_albert_graph(n, m=2).to_directed()
    else:
        for i in range(n):
            G.add_edge(i, (i + 1) % n)
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < p:
                    G.add_edge(i, j)
    return G

def load_x_thread_graph(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    G = nx.DiGraph()
    for edge in data:
        u = edge.get("from")
        v = edge.get("to")
        if u and v:
            G.add_edge(u, v)
    if not nx.is_strongly_connected(G) and G.number_of_nodes() > 0:
        largest = max(nx.strongly_connected_components(G), key=len)
        nodes = list(largest)[:min(5, len(largest))]
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
    return G

# ────────────────────────────────────────────────
# Invariant checking
# ────────────────────────────────────────────────

class InvariantChecker:
    def __init__(self, G, dom_thresh):
        self.G = G
        self.dom_thresh = dom_thresh

    def check_closure(self):
        sinks = [n for n in self.G.nodes if self.G.out_degree(n) == 0]
        violated = bool(sinks)
        return violated, len(sinks)

    def check_mutual(self):
        zero_in = sum(1 for _, d in self.G.in_degree() if d == 0)
        zero_out = sum(1 for _, d in self.G.out_degree() if d == 0)
        orphans = zero_in + zero_out
        not_connected = not nx.is_strongly_connected(self.G)
        violated = orphans > 0 or not_connected
        return violated, orphans

    def check_non_domination(self):
        indeg_values = [d for _, d in self.G.in_degree()]
        if not indeg_values:
            return False, 0, 0.0, False

        max_indeg = max(indeg_values)
        loose_viol = max_indeg > self.dom_thresh

        gini_val = self._gini_coefficient(indeg_values)

        between = nx.betweenness_centrality(self.G)
        high_indeg_nodes = [n for n, d in self.G.in_degree() if d > self.dom_thresh // 2]
        primacy_trigger = any(between.get(n, 0) > 0.3 for n in high_indeg_nodes) if high_indeg_nodes else False

        tight_viol = gini_val > 0.6 or primacy_trigger
        violated = loose_viol or tight_viol

        return violated, max_indeg, gini_val, tight_viol

    @staticmethod
    def _gini_coefficient(values):
        if not values:
            return 0.0
        arr = np.sort(np.array(values))
        n = len(arr)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * arr) / (n * np.sum(arr))) if np.sum(arr) > 0 else 0.0

# ────────────────────────────────────────────────
# Metrics computation
# ────────────────────────────────────────────────

def compute_metrics(G, add_attributes=False):
    sccs = list(nx.strongly_connected_components(G))
    num_sccs = len(sccs)
    non_trivial_nodes = sum(len(scc) for scc in sccs if len(scc) > 1)
    frac_non_triv = non_trivial_nodes / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    cond = nx.condensation(G)
    cond_edges = cond.number_of_edges()
    sources = sum(1 for _, d in cond.in_degree() if d == 0)
    dag_height = nx.dag_longest_path_length(cond) if cond.number_of_nodes() > 0 else 0

    try:
        longest_path = nx.dag_longest_path_length(G)
    except nx.NetworkXUnfeasible:
        longest_path = dag_height * (G.number_of_nodes() / max(num_sccs, 1))  # heuristic

    metrics = {
        'num_sccs': num_sccs,
        'frac_non_triv': frac_non_triv,
        'cond_edges': cond_edges,
        'sources': sources,
        'dag_height': dag_height,
        'longest_path': longest_path
    }

    if add_attributes:
        indeg = dict(G.in_degree())
        outdeg = dict(G.out_degree())
        between = nx.betweenness_centrality(G)
        scc_map = {node: len(c) for c in sccs for node in c}
        for node in G:
            G.nodes[node]['in_degree'] = indeg.get(node, 0)
            G.nodes[node]['out_degree'] = outdeg.get(node, 0)
            G.nodes[node]['betweenness'] = between.get(node, 0)
            G.nodes[node]['scc_size'] = scc_map.get(node, 1)
            G.nodes[node]['is_orphan'] = indeg.get(node, 0) == 0 or outdeg.get(node, 0) == 0

    return metrics

# ────────────────────────────────────────────────
# GraphML export
# ────────────────────────────────────────────────

def export_to_graphml(G, filename):
    print(f"Exporting to {filename}...")
    nx.write_graphml(G, filename)

# ────────────────────────────────────────────────
# Batch simulation
# ────────────────────────────────────────────────

class BatchSimulator:
    def __init__(self, args):
        self.args = args

    def run_single_p(self, p):
        totals = {
            'violation': 0,
            'max_indeg': 0.0,
            'gini': 0.0,
            'primacy': 0.0,
            'orphans': 0.0,
            'num_sccs': 0.0,
            'frac_non_triv': 0.0,
            'cond_edges': 0.0,
            'sources': 0.0,
            'dag_height': 0.0,
            'longest_path': 0.0
        }
        for run in range(self.args.runs):
            random.seed(self.args.seed + run)
            G = generate_synthetic_graph(self.args.n, p, self.args.seed + run, self.args.prefer_attach)
            checker = InvariantChecker(G, self.args.n)
            c_viol, _ = checker.check_closure()
            m_viol, orphans = checker.check_mutual()
            nd_viol, max_indeg, gini, primacy = checker.check_non_domination()

            violation = 1 if c_viol or m_viol or nd_viol else 0

            totals['violation'] += violation
            totals['max_indeg'] += max_indeg
            totals['gini'] += gini
            totals['primacy'] += primacy
            totals['orphans'] += orphans

            metrics = compute_metrics(G)
            totals['num_sccs'] += metrics['num_sccs']
            totals['frac_non_triv'] += metrics['frac_non_triv']
            totals['cond_edges'] += metrics['cond_edges']
            totals['sources'] += metrics['sources']
            totals['dag_height'] += metrics['dag_height']
            totals['longest_path'] += metrics['longest_path']

        averages = {k: v / self.args.runs for k, v in totals.items()}
        return averages

    def run(self):
        print("|   p   | Viol Prob | Max In-Deg | In Gini | Primacy Risk | Orphans | # SCCs | Frac Non-Triv | Cond Edges | Sources | DAG Height | Longest Path |")
        print("|-------|-----------|------------|---------|--------------|---------|--------|---------------|------------|---------|------------|--------------|")

        num_steps = int((self.args.p_max - self.args.p_min) / self.args.p_step) + 1
        for i in range(num_steps):
            p = round(self.args.p_min + i * self.args.p_step, 2)
            avgs = self.run_single_p(p)
            viol_prob = avgs['violation']
            avg_primacy_pct = avgs['primacy'] * 100

            print(f"| {p:5.2f} | {viol_prob:9.3f} | {avgs['max_indeg']:10.1f} | {avgs['gini']:7.3f} | {avg_primacy_pct:12.1f}% | {avgs['orphans']:7.1f} | {avgs['num_sccs']:6.1f} | {avgs['frac_non_triv']:13.3f} | {avgs['cond_edges']:10.1f} | {avgs['sources']:7.1f} | {avgs['dag_height']:10.1f} | {avgs['longest_path']:12.1f} |")


# ────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CORNet Simulator")
    parser.add_argument('--n', type=int, default=20, help='Number of nodes (default: 20)')
    parser.add_argument('--p', type=float, default=0.02, help='Edge probability (default: 0.02)')
    parser.add_argument('--samples', type=int, default=500, help='MC samples for density (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--force-violation', action='store_true', help='Force violation mode (n=50, p=0.05)')
    parser.add_argument('--k', type=int, default=6, help='Term size bound k (default: 6)')
    parser.add_argument('--max-depth', type=int, default=3, help='Max derivation depth (default: 3)')
    parser.add_argument('--batch', action='store_true', help='Run batch simulation over p values')
    parser.add_argument('--p_min', type=float, default=0.01, help='Min p for batch (default: 0.01)')
    parser.add_argument('--p_max', type=float, default=0.20, help='Max p for batch (default: 0.20)')
    parser.add_argument('--p_step', type=float, default=0.01, help='Step for p in batch (default: 0.01)')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs per p in batch (default: 30)')
    parser.add_argument('--prefer-attach', action='store_true', help='Use Barabási-Albert preferential attachment')
    parser.add_argument('--x-thread', type=str, default=None, help='Path to JSON file with X thread edges')
    parser.add_argument('--export-graphml', type=str, default=None, help='Export graph to GraphML file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.force_violation:
        args.n = 50
        args.p = 0.05

    if args.x_thread:
        G = load_x_thread_graph(args.x_thread)
    else:
        G = generate_synthetic_graph(args.n, args.p, args.seed, args.prefer_attach)

    checker = InvariantChecker(G, args.n)
    c_viol, sinks = checker.check_closure()
    m_viol, orphans = checker.check_mutual()
    nd_viol, max_indeg, gini, primacy = checker.check_non_domination()

    invariants_preserved = not c_viol and not m_viol and not nd_viol
    emotion_render = 'positive (calm)' if invariants_preserved else 'negative (distress)'

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Closure violated: {c_viol} (sinks: {sinks})")
    print(f"Mutual instantiation violated: {m_viol} (orphans: {orphans})")
    print(f"Non-domination violated: {nd_viol} (max indeg: {max_indeg}, Gini: {gini:.3f}, Primacy: {primacy})")
    print(f"Invariants preserved: {invariants_preserved}")
    print(f"Emotion rendering: {emotion_render}")

    metrics = compute_metrics(G, add_attributes=args.export_graphml is not None)
    print(f"Metrics: {metrics}")

    if args.export_graphml:
        export_to_graphml(G, args.export_graphml)

    if args.batch:
        BatchSimulator(args).run()

    if args.samples > 0:
        # Density estimation
        op = Function('op', commutative=False)
        e = symbols('e')
        a, b, c = Wild('a'), Wild('b'), Wild('c')
        relations = [
            Eq(op(e, a), a),
            Eq(op(a, e), a),
            Eq(op(a, b), op(b, a)),
            Eq(op(a, op(b, c)), op(op(a, b), c)),
            Eq(op(a, a), a)
        ]
        pres = EquationalPresentation([op, e], relations, associative_ops=[op])
        variables = symbols('v1 v2 v3 v4 v5') + (e,)
        p_hat, interval = estimate_invariants_mc(
            pres, variables, k=args.k, samples=args.samples, max_depth=args.max_depth
        )
        d = len(relations)
        density = p_hat / d
        print(f"\nInvariant proportion (p_hat): {p_hat:.4f}")
        print(f"95% Wilson CI: ({interval[0]:.4f}, {interval[1]:.4f})")
        print(f"Approximate density (p_hat / {d} axioms): {density:.4f}")