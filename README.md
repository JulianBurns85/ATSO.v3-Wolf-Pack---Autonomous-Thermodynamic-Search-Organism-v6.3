# Autonomous Thermodynamic Search Organism (V6.3)

An online-adapting, homeostatically regulated Markov password policy auditor and stochastic search organism optimized for multi-core CPU execution via Numba (`@njit(parallel=True)`) with a two-way 3D WebGL Command & Control HUD.

---

## Key Architectural Pillars

1. **3D Positional Stigmergic Tensor (`main_engine.py`)**
   - Transitions are conditioned on exact character index `[PASS_MAX - 1, 96, 96]` with explicit `<EOS>` (End-of-String) tokenization at index `0` to support variable-length candidates (`4` to `8` characters).
   - Eliminates cross-position path interference between prefixes, root bodies, numeric suffixes, and trailing symbols.

2. **Positional Character-Class Diffusion**
   - When a candidate hit verifies a transition into a semantic character class (such as ASCII digits `0-9` or trailing symbols `!@#$`), sympathetic weight diffuses across adjacent class members at that specific positional slice to bridge unseeded structural permutations.

3. **Active Adversarial Mutation Engine**
   - Continuously applies phonetic substitution rules (`e -> 3`, `a -> @`, `i -> 1`, `o -> 0`, `s -> $`, `g -> 9`) and symbol suffixes to high-probability trajectories, enabling rapid evaluation of predictable l33t-speak policy mutations.

4. **Thermodynamic Entropy Watchdog (`entropy_watchdog.py`)**
   - Continuously computes active-row Shannon entropy across tensor slices:
     $$H(R_i) = -\sum_{j} P(x_{i,j}) \log_2 P(x_{i,j})$$
   - Prevents competitive exclusion and mode collapse by applying power-law annealing ($W^{0.78}$) whenever active rows drop below the homeostatic floor (`1.5 bits`) and relaxing active weights toward the base prior (`0.96 * active + 0.04 * prior`).

5. **Two-Way 3D WebGL Cyber Command HUD (`index.html` & `telemetry_server.py`)**
   - **Downlink (25 FPS Telemetry):** Streams normalized 95x95 tensor projections, active Shannon entropy sparklines, and a real-time solved hash ticker over WebSockets (`ws://localhost:8765`).
   - **Uplink (Interactive Policy Auditor & Slice Selector):** Allows operators to isolate individual positional tensor slices (`MAX`, `P0->1` .. `P5->6`) on the 3D terrain and inject custom comma-separated password policies directly from the browser.

---

## Empirical Benchmark Results (12 Correlated SHA-256 Targets, 20s Limit)

| Engine Architecture | Targets Solved | Time Elapsed | Candidates Evaluated | Efficiency vs. Static |
| :--- | :--- | :--- | :--- | :--- |
| **Static 2D Markov Baseline** | 05 / 12 | 20.12s (Timeout) | 3,540,000+ | 1.0x (Stalled) |
| **V1: Exact Stigmergy Only** | 06 / 12 | 20.00s (Last: 0.38s) | 9,274,760 | Stalled on unseeded digits |
| **V2: + Class Diffusion** | 08 / 12 | 20.00s (Last: 0.55s) | 8,427,600 | Prefix lockout on 3rd family |
| **V3: 2D Regulated Lattice** | **12 / 12** | **0.92s - 1.51s** | **407,773 - 689,988** | **23.8x fewer evaluations** |
| **V4: 3D Positional Tensor** | **12 / 12** | **0.25s (Sweep)** | **86,678** | **40.8x fewer evaluations (7.96x over V3)** |
| **V6.3: Variable-Length + HUD** | **6 / 6 per Wave** | **0.16s - 5.60s** | **Dynamic `<EOS>`** | **Full L33t/Symbol Policy Auditing** |

---

## Quick Start

    # 1. Run the 3-way stopwatch benchmark (Static vs. V3 2D vs. V4 3D Positional)
    python benchmark_fair.py

    # 2. Launch the V6.3 Interactive Command & Control Engine
    python main_engine.py

    # 3. Open index.html in any modern browser to view the 3D lattice,
    #    toggle positional slices, and launch custom policy audits.
