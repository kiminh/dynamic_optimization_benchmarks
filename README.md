# dynamic-optimization-benchmarks

The goal of this package is to provide a unified way to compare dynamic optimization algorithms. The initial focus is on dynamic matching, with applications to ``Kidney Exchange`` and ``Carpooling``.

The code is divided into two parts:
- The environments, which implement the specifics of the dynamic optimization problems.
- The algorithms, which implement various ideas from the dynamic optimization literature.

## 1. Quickstart examples

### Running code for the figures in [``Maximizing Efficiency in Dynamic Matching Markets``](https://arxiv.org/abs/1803.01285)
Run ``main.py``.

### Running algorithms with custom parameters
Edit file ``generate_sim_plan``.
Choose a random seed.
Create a folder ``results/resultsxx`` where xx is the number of the random seed.
Run ``main.py``.

## 2. Environments

### Matching
The matching environment corresponds to the problem described in the paper:
The state space is an edge-weighted graph, which evolves over time as vertices
arrive, are matched or depart.

At each time period, the environment receives a matching from the online algorithm.
It then performs the following steps:
 - Removes matched vertices from the graph,
 - Computes the value of matched edges,
 - Computes departures among unmatched vertices,
 - Samples new vertex arrivals.

## 3. Algorithms

### Greedy
The ``greedy`` algorithm returns a maximum-weight matching over the current state
graph at each time step

### Batching
The ``batching`` algorithm returns a maximum-weight matching every ``b`` time periods.
It returns an empty matching the rest of the time.

### Re-Opt
The Re-Opt algorithm computes a maximum-weight-matching every time period.
It then returns an edge in the matching if and only if one of the vertices adjacent
to that edge is about to depart (or ``critical``).
When information about critical vertices is not available, it is estimated when
vertices have been waiting for some number of time periods, which is controlled
by a ``patience`` parameter.

### 𝛼-Re-Opt.
This works similarly to Re-Opt, except that the weights of ``critical`` vertices are now increased by a factor 𝛼 ∈ [1,2] before running the max-weight-matching algorithm.
