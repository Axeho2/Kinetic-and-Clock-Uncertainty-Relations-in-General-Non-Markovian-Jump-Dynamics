# Kinetic and Clock Uncertainty Relations in General Non-Markovian Jump Dynamics

Simulation and figure-generation code accompanying the paper *Kinetic and Clock Uncertainty Relations in General Non-Markovian Jump Dynamics*.

The simulations use an eight-state molecular model with a switching catalytic gate to illustrate the hierarchy of timing information retained by unordered transition counts (`N`), ordered event sequences (`Sigma`), and time-stamped trajectories (`X`).

## Contents

- **`cur_hierarchy/`** — Gillespie simulations and plots of the uncertainty-bound hierarchy (`simulate.py`, `cur_hierarchy.tex`).
- **`barrier_advantage/`** — Simulations and plots showing the advantage of event-order information as the gate-switching barrier increases (`sample_barrier_advantage.py`, `barrier_advantage.tex`).

## Requirements

Python 3 with `numpy`, `scipy`, and `numba`; LuaLaTeX with `pgfplots` for compiling the figures.

## Usage

Run the Python script in each directory to generate its `.dat` files, then compile the corresponding `.tex` file.

For example, in `cur_hierarchy/`:

```bash
python simulate.py --preset validation
lualatex cur_hierarchy.tex
```

In `barrier_advantage/`:

```bash
python sample_barrier_advantage.py
lualatex barrier_advantage.tex
```

See `--help` for simulation parameters. The default simulations can be computationally intensive.

## License

[CC0 1.0 Universal](LICENSE).
