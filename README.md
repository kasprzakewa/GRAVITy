# GRAVITy

**Gravitational Restricted Astrodynamics Variational Integration Toolkit**

A comprehensive toolkit for numerical integration and trajectory optimization in the Planar Circular Restricted Three-Body Problem (PCR3BP), specifically configured for the Earth-Moon system.

## Overview

The project evaluates the effectiveness of classical Newtonian schemes against symplectic methods based on Hamiltonian formalism. The toolkit provides:

* **Newtonian Integrators**: High-order explicit schemes (Verner, Dormand-Prince, Adams-Bashforth) for efficient short-term propagation.
* **Symplectic Integrators**: Implicit methods (Gauss, Lobatto) designed to preserve the geometric structure and energy stability in long-term simulations of non-separable Hamiltonian systems.
* **Trajectory Optimization**: Algorithm for determining low-energy transfers using stable and unstable manifolds associated with Lyapunov orbits around the $L_1$ point.

## Mathematical Background

### PCR3BP Equations of Motion

The system describes the motion of a massless test particle under the influence of two primary masses, $m_1$ and $m_2$, in a rotating reference frame.

**Newtonian Formulation**

In the rotating frame, the equations of motion are derived directly from Newton's laws, explicitly accounting for the Coriolis force, centrifugal force and the gravitational pull of the two primaries:

$$
\ddot{x} - 2\dot{y} = x - \frac{1-\mu}{r_1^3}(x+\mu) - \frac{\mu}{r_2^3}(x - 1 + \mu)
$$

$$
\ddot{y} + 2\dot{x} = y - \frac{1-\mu}{r_1^3}y - \frac{\mu}{r_2^3}y
$$

where the distances to the primary bodies are defined as:

$$
r_1 = \sqrt{(x+\mu)^2 + y^2}, \quad r_2 = \sqrt{(x - 1 + \mu)^2 + y^2}
$$

**Hamiltonian Formulation**

The system is described by a non-separable Hamiltonian function $H(q, p)$, expressed using the effective potential $\bar{U}$:

$$
H(x, y, p_x, p_y) = \frac{1}{2}((p_x + y)^2 + (p_y - x)^2) + \bar{U}(x,y)
$$

where generalized momenta are $p_x = \dot{x} - y$ and $p_y = \dot{y} + x$, and the **effective potential** is defined as:

$$
\bar{U}(x,y) = -\frac{1-\mu}{r_1} - \frac{\mu}{r_2} - \frac{1}{2}(x^2 + y^2)
$$

**Energy Integral (Jacobi Constant)**

The system possesses a conserved quantity known as the Jacobi constant ($C$), which is related to the total energy ($E$) and serves as the primary metric for evaluating numerical stability:

$$
C = -2E = -2\bar{U}(x,y) - (\dot{x}^2 + \dot{y}^2)
$$

### Key Features

- **Newtonian Methods**: Euler, Midpoint, RK4, Verner9 (9th order), Dormand-Prince 8 (7th/8th order), Adams-Bashforth 5
- **Hamiltonian Methods**: Gauss-Legendre collocation (symplectic), Lobatto IIIA-IIIB pairs, Explicit Partitioned Runge-Kutta (EPRK)
- **Energy Conservation Analysis**: Automated testing across multiple energy levels and integration parameters
- **Trajectory Optimization**: HEO-to-Lyapunov orbit transfers via invariant manifolds
- **Visualization Tools**: Heatmaps, comparison plots, effective potential surfaces
- **Automated Benchmarking**: Performance metrics, energy drift analysis, method ranking

## Installation

### Prerequisites

- **Julia** 1.10.0 or higher (tested with Julia 1.10.0)

### Quick Install

```bash
git clone https://github.com/kasprzakewa/GRAVITy.git
cd GRAVITy
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Usage

### Running Tests

Test all numerical integration methods:
```bash
julia --project=. GRAVITy.jl test
```

Test only Newtonian methods (Euler, RK4, Vern9, DP8, AB5):
```bash
julia --project=. GRAVITy.jl test --newtonian
```

Test only Hamiltonian methods (Gauss, Lobatto, EPRK):
```bash
julia --project=. GRAVITy.jl test --hamiltonian
```

### Trajectory Optimization

Run invariant manifold trajectory optimization example:
```bash
julia --project=. GRAVITy.jl trajectory
```

### Analysis and Visualization

Generate plots and analysis from test results:
```bash
julia --project=. GRAVITy.jl analyze
```

This creates comparison plots and heatmaps in `results/plots/`.

### Help

Display all available commands:
```bash
julia --project=. GRAVITy.jl help
```

## Project Structure

```
GRAVITy/
├── GRAVITy.jl                 # Main entry point (CLI)
├── src/                       # Core modules
│   ├── CommonUtils.jl         # CR3BP utilities, test cases
│   ├── NewtonianMethods.jl    # Classical ODE integrators
│   ├── HamiltonianMethods.jl  # Symplectic integrators
│   └── TrajectoryOptimization.jl  # Manifold computations
├── utils/                     # Analysis and testing scripts
│   ├── run_tests.jl           # Comprehensive test runner
│   ├── run_trajectory_optimization.jl  # Optimization example
│   └── analyze_and_plot_results.jl     # Visualization
├── results/                   # Output files and plots
└── example_data/              # Sample trajectory data
```

## Test Cases

The toolkit tests methods across three energy level regimes in the Earth-Moon PCR3BP:

1. **E < E_L1**: Below L1 Lagrange point energy (bounded lunar orbits)
2. **E_L1 < E < E_L2**: Between L1 and L2 energies (transition orbits)
3. **E_L4 < E**: Above L4/L5 energy (high-energy orbits)

Each test case varies:
- **Time step (dt)**: 0.001, 0.01, 0.1
- **Integration time (T)**: 10, 50, 100 dimensionless units

## Output

### Results Files

- `results/newtonian_methods/newtonian_methods_results.csv` - Newtonian methods benchmark data
- `results/hamiltonian_methods/hamiltonian_methods_results.csv` - Hamiltonian methods benchmark data
- `results/pcr3bp_results.csv` - Combined results

### Plots

Generated in `results/plots/[method_type]/`:
- **Heatmaps**: Energy drift and computation time across parameter space
- **Fixed T plots**: Energy conservation vs. time step for fixed integration time
- **Fixed dt plots**: Energy conservation vs. integration time for fixed time step

## Example Output

Running tests produces detailed summaries:
```
NEWTONIAN METHODS SUMMARY
========================================
Verner's 9th order method:
  E < E_L1: Best |ΔE| = 2.44e-15 (dt=0.01, time=2.88s)
  
Dormand-Prince 7th order method:
  E < E_L1: Best |ΔE| = 6.88e-15 (dt=0.01, time=2.63s)

OVERALL RANKING BY ENERGY CONSERVATION
========================================
1. Verner's 9th order method (E < E_L1): |ΔE| = 2.44e-15
2. Verner's 9th order method (E_L1 < E < E_L2): |ΔE| = 2.66e-15
3. Dormand-Prince 7th order method (E_L1 < E < E_L2): |ΔE| = 6.44e-15
...
```

## License

See [LICENSE](LICENSE) file for details.