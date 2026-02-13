#!/usr/bin/env python3
"""
Observation Network Tool – Bounded relational invariant checks & density estimation
Witness to closure, non-domination, and rendering from Tonnel's papers.
"""
import argparse
import networkx as nx
import random
from sympy import symbols, Eq, Function, Wild
# ────────────────────────────────────────────────
# Density estimation core
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
class Presentation:
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
def estimate_invariants_mc(presentation, variables, k=6, samples=500, max_depth=3):
    ops = [g for g in presentation.generators if callable(g)]
    hits = 0
    for _ in range(samples):
        t1 = presentation.normalize(random_term(variables, ops, k))
        t2 = presentation.normalize(random_term(variables, ops, k))
        if t1 != t2 and presentation.derive(Eq(t1, t2), max_depth=max_depth):
            hits += 1
    p_hat = hits / samples
    return p_hat, wilson_interval(p_hat, samples)
# ────────────────────────────────────────────────
# Main CLI entry point
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Observation Network Tool – Bounded invariant checks & density estimation"
    )
    parser.add_argument('--n', type=int, default=20,
                        help='Number of relations/nodes (default: 20)')
    parser.add_argument('--p', type=float, default=0.02,
                        help='Probability of random extra edges (default: 0.02)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Monte-Carlo samples for density (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--force-violation', action='store_true',
                        help='Force violation mode (n=50, p=0.05)')
    parser.add_argument('--k', type=int, default=6,
                        help='Term size bound k (default: 6)')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Max derivation depth n (default: 3)')
    args = parser.parse_args()
    # Override for violation demonstration
    if args.force_violation:
        args.n = 50
        args.p = 0.05
    random.seed(args.seed)
    # Build network
    G = nx.DiGraph()
    for i in range(args.n):
        G.add_edge(i, (i + 1) % args.n)
    for i in range(args.n):
        for j in range(args.n):
            if i != j and random.random() < args.p:
                G.add_edge(i, j)
    # Evaluate invariants
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    closure_violated = bool(sinks)
    mutual = len(G.edges) > 0
    in_deg = dict(G.in_degree())
    domination_threshold = args.n // 4
    domination_potential = max(in_deg.values()) > domination_threshold
    invariants_preserved = not closure_violated and not domination_potential
    emotion_render = 'positive (calm)' if invariants_preserved else 'negative (distress)'
    print(f"Nodes: {args.n}, Edges: {len(G.edges)}")
    print(f"Closure violated: {closure_violated}")
    print(f"Mutual instantiation: {mutual}")
    print(f"Domination potential: {domination_potential} (threshold > {domination_threshold})")
    print(f"Invariants preserved: {invariants_preserved}")
    print(f"Emotion rendering: {emotion_render}")
    cycles = list(nx.simple_cycles(G))[:10]
    print(f"Sample cycles (up to 10): {cycles}")
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
    pres = Presentation([op, e], relations, associative_ops=[op])
    variables = symbols('v1 v2 v3 v4 v5') + (e,)
    p_hat, interval = estimate_invariants_mc(
        pres, variables, k=args.k, samples=args.samples, max_depth=args.max_depth
    )
    d = len(relations)
    density = p_hat / d
    print(f"\nInvariant proportion (p_hat): {p_hat:.4f}")
    print(f"95% Wilson CI: ({interval[0]:.4f}, {interval[1]:.4f})")
    print(f"Approximate density (p_hat / {d} axioms): {density:.4f}")
if __name__ == '__main__':
    main()