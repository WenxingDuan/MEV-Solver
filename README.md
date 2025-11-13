# MEV solver

This repository provides a Python research-focused implementation of two routing engines for DEX-like markets and a small benchmarking/plotting toolchain. It is designed to be simple to run end-to-end: load benchmark instances, execute both solvers, write JSON summaries, and generate comparison figures (PNG).

All non-Python assets from the original project have been removed so this repository focuses on the core algorithms and scripts.

**Key Capabilities**
- GeneticRouterEngine (NSGA-II) for multi-objective routing
- DualDecompositionOptimizer (Bellman–Ford baseline) for deterministic routing
- Batch evaluation across instances and rich metrics JSON
- 2×2 comparison figure showing effectiveness, robustness, and cost

**Requirements**
- Python ≥ 3.9
- Optional plotting: `python3 -m pip install matplotlib`

**Dataset**
- Benchmarks live under `benchmarks/instances/` and are local snapshots adapted from the public Alpha‑Router project.

**Deterministic Seeding**
- All tools share a xorshift32 RNG (`python/seeded_random.py`); repeated runs with the same `--seed` reproduce identical evolutionary trajectories.

**How To Run (Quick Start)**
- GA solver → prints a single‑instance JSON to stdout
  - `python3 python/run_ga.py --instance benchmarks/instances/sample_instance.json --seed 1337`
- Deterministic baseline → prints a single‑instance JSON to stdout
  - `python3 python/run_dual.py --instance benchmarks/instances/sample_instance.json`
- Batch evaluate both solvers on all instances → writes a summary JSON
  - `python3 python/evaluate_engines.py --instances-dir benchmarks/instances --output benchmarks/results/ga_vs_dual_summary.json`
- Plot a 2×2 comparison figure → writes PNG
  - `python3 python/plot_ga_vs_dual.py --summary benchmarks/results/ga_vs_dual_summary.json --output figures/ga_vs_dual.png`

**Output Files**
- Single GA run (stdout): best chromosome + Pareto front JSON (see “GA JSON Schema”)
- Single deterministic run (stdout): best path JSON (see “Deterministic JSON Schema”)
- Batch summary: `benchmarks/results/ga_vs_dual_summary.json`
- Figure: `figures/ga_vs_dual.png`

**CLI Options (Common)**
- `--instance`: path to a single benchmark JSON (for single‑run tools)
- `--instances-dir`: directory with multiple benchmarks (for batch)
- `--seed`: deterministic RNG seed (GA and batch)
- `--amount`: override order size in token units (otherwise read from instance metadata)

**GA JSON Schema (Single Run)**
- Top‑level
  - `seed`: RNG seed used
  - `config`: GA parameters (population, generations, mutation/crossover, limits)
  - `amount`: order size consumed by the simulator
  - `generations`: completed NSGA‑II generations
  - `best`: the selected chromosome from Pareto rank 1
  - `paretoFront`: all rank‑1 chromosomes
- Chromosome fields
  - `paths`: list of token hops; each hop has `fromToken`, `toToken`, `poolAddress`, `protocol`
  - `splitRatios`: normalized path allocations (sum to 1)
  - `fitness`: `surplus` (quote‑token output), `gasUnits` (base + per‑hop gas), `slippage` (mid‑price gap)
  - `rank`, `crowdingDistance`: NSGA‑II rank and diversity (boundary points may show `null` crowding)

**Deterministic JSON Schema (Single Run)**
- Top‑level
  - `method`: `DETERMINISTIC`
  - `expectedSurplus`: output from replaying the extracted path in the simulator
  - `gasUnits`: base + per‑hop gas estimate
  - `paths`: a single best path in the same hop format as GA
  - `splitRatios`: `[1.0]` for a single‑path allocation
  - `pathLength`: number of hops in the route
  - `slippage`: cumulative mid‑price gap along the route

**Batch Summary JSON Schema** (`benchmarks/results/ga_vs_dual_summary.json`)
- Per‑instance fields
  - `instance`, `orderSize`, `orderSizeClass`, `fragmentationLabel`, `ammDiversity`, `gasRegime`
  - `marketCount`, `tokenCount`, `fragmentationScore` (complexity proxies)
  - `ga`: `{surplus, gasUnits, slippage, netSurplus, runtimeMs, pathCount, avgHopCount, outputPerGas, splitRatios, rank, generations}`
  - `deterministic`: `{surplus, gasUnits, runtimeMs, pathLength, paths, slippage, outputPerGas}`
  - `surplusDelta`, `slippageDelta`, `outputPerGasDelta`, `runtimeRatio`

**Figure (2×2 Dashboard)**
- Top‑left: surplus (quote‑token output) vs. fragmentation (effectiveness)
- Top‑right: slippage vs. fragmentation (robustness to price impact)
- Bottom‑left: runtime vs. market count (computational cost)
- Bottom‑right: output‑per‑gas vs. market count (capital efficiency); GA marker size encodes path count

**Project Layout**
- `python/genetic_router_engine.py` — NSGA‑II solver (path‑set chromosome + split ratios; AMM replay fitness)
- `python/dual_decomposition_optimizer.py` — deterministic Bellman–Ford baseline (negative‑cycle detection)
- `python/path_simulator.py` — common AMM simulator and market map helpers
- `python/token_utils.py` — token decimals and normalization utilities
- `python/seeded_random.py` — xorshift32 RNG used across tools
- `python/run_ga.py` — CLI for a single GA run (prints JSON)
- `python/run_dual.py` — CLI for a single deterministic run (prints JSON)
- `python/evaluate_engines.py` — batch evaluator; writes `benchmarks/results/ga_vs_dual_summary.json`
- `python/plot_ga_vs_dual.py` — plotter; writes `figures/ga_vs_dual.png`
- `python/compare_runs.py` — compare two GA JSON outputs (`--lhs`, `--rhs`) for exact parity
- `benchmarks/instances/` — input datasets (JSON snapshots)
- `benchmarks/results/` — generated summaries
- `figures/` — generated figures

**Reproducibility Tips**
- Pin the seed (`--seed`) and GA hyperparameters when comparing runs.
- Use `--amount` to sweep order sizes without editing instances.
- Run `evaluate_engines.py` before plotting to refresh the summary JSON.

