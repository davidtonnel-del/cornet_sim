# CORNet Simulator  
**Closed Observational Relationalism Network Analysis Tool**

A bounded, syntactic simulator for **closed observational networks** (R) based on the relational framework developed by David Gérard Tonnel (2025–2026).  

The tool checks relational invariants, detects collapse via graph predicates, estimates invariant density, simulates density-dependent transitions, and diagnoses scale asymmetry — all internally to the agent's observation network, with no external frames or new ontological commitments.

### Core Theoretical Foundations (from the papers)

- **Primitive**: Observation relation (observer ↔ observed) — mutually instantiated, no independent roles.
- **Role continuation** ri → rj enforces **recursive closure** — no terminals, no external reference frames.
- **Emergent features**:
  - Reachability (temporal ordering)
  - Incomparability (spatial separation)
  - Stable patterns (subgraphs with multiple structure-preserving embeddings)
- **Relational invariants** required for coherent persistence:
  - Mutual instantiation (co-instantiation without primacy)
  - Non-domination (preservation without authorship erasure)
  - Closure (recursive, no terminals)
- **Emotion** as orthogonal, lossy phenomenological rendering of invariant evaluation outcome ({Preserve → calm, Violate → distress}), not primitive or epistemically authoritative.
- **Scale asymmetry** — higher-scale systems constrain lower-scale dynamics non-locally (coherence signatures) without reciprocal representation; intervention collapses distinction via domination.
- **Bounded collapse detection** — syntactic predicates over finite R (sinks for closure, in-degree/Gini/betweenness for non-domination, orphans/asymmetry for mutual).

### Features

- Synthetic graph generation: base cycle + uniform random or preferential attachment
- Real X/Twitter thread ingestion (`--x-thread <json>`) — builds directed reply/quote graph
- Invariant checks: closure, mutual (orphans + connectivity), non-domination (loose max in-degree + tighter Gini + betweenness primacy)
- Stable pattern proxy: SCC metrics (# SCCs, fraction in non-trivial components)
- Scale asymmetry: condensation DAG (edges, sources, DAG height, longest path)
- Density estimation: Monte-Carlo on fixed equational presentation
- Batch mode: density transitions over p range, with violation probability and primacy risk
- Modular design: easy to extend invariants, generators, or output formats

## New Feature: GraphML Export

Visualize networks in Gephi, Cytoscape, yEd, etc.

```bash
# Export synthetic graph after analysis
python cornet_sim.py --n 100 --p 0.05 --export-graphml output.graphml

Example Datasets

The repository includes small experimental thread graphs for testing collapse detection and scale asymmetry diagnostics:

examples/threads/


Example usage:

python cornet_sim.py --x-thread examples/threads/censorship_thread.json
python cornet_sim.py --x-thread examples/threads/filterbubble_thread.json


These datasets are illustrative network abstractions for structural experimentation only.

# Export real X thread graph
python cornet_sim.py --x-thread thread.json --export-graphml x_thread.graphml

### Requirements

```text
networkx==3.3
sympy==1.13.2
numpy==1.26.4
Install with:
Bashpip install -r requirements.txt
Usage
Bash# Basic synthetic run (preserving regime)
python cornet_sim.py --n 50 --p 0.02 --seed 42

# Force violation example
python cornet_sim.py --n 50 --p 0.05 --seed 42

# Batch density transitions (phase-like collapse analysis)
python cornet_sim.py --batch --n 100 --p_min 0.01 --p_max 0.20 --p_step 0.01 --runs 30 --seed 42

# Preferential attachment (influencer-heavy social graph)
python cornet_sim.py --n 200 --prefer-attach --batch --runs 20

# Analyze real X thread (after exporting replies to JSON)
python cornet_sim.py --x-thread thread_data.json
Example Batch Output (excerpt)
text|   p   | Viol Prob | Max In-Deg | In Gini | Primacy Risk | Orphans | # SCCs | Frac Non-Triv | Cond Edges | Sources | DAG Height | Longest Path |
|-------|-----------|------------|---------|--------------|---------|--------|---------------|------------|---------|------------|--------------|
|  0.01 |     0.000 |        3.2 |   0.312 |          0.0% |     0.0 |    1.0 |         1.000 |        0.0 |     1.0 |        0.0 |         49.0 |
|  0.05 |     0.000 |        5.9 |   0.421 |          0.0% |     0.0 |    1.0 |         1.000 |        0.0 |     1.0 |        0.0 |         49.0 |
...
Scope & Limits

All analysis is internal to R (agent's own relations) — no external reality modeling.
Non-domination proxied loosely (max in > n/4) and tightly (Gini >0.6 or high-betweenness primacy).
Stable patterns via SCCs (no exponential cycle enumeration).
Rendering orthogonal to evaluation (Preserve → calm, Violate → distress).
Density estimation uses fixed 5-axiom presentation.
No psychological, moral, or completeness claims — bounded, syntactic, admissibility-preserving.

MIT License. Research and demonstration purposes only.
See the referenced papers for full theoretical justification.