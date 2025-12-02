# GRAVITy - CR3BP Comprehensive Testing Suite

Comprehensive numerical methods testing and trajectory optimization for the Circular Restricted Three-Body Problem (CR3BP) in the Earth-Moon system.

## Quick Start

```bash
# Run all numerical tests
julia GRAVITy.jl test

# Run only Newtonian methods
julia GRAVITy.jl test --newtonian

# Run only Hamiltonian methods
julia GRAVITy.jl test --hamiltonian

# Run trajectory optimization
julia GRAVITy.jl trajectory

# Show help
julia GRAVITy.jl help
```

## Usage

### Main Command Line Interface

```bash
julia GRAVITy.jl [command] [options]
```

**Commands:**
- `test` - Run numerical methods tests
- `trajectory` - Run trajectory optimization
- `help` - Show help message

**Test Options:**
- `--newtonian` - Run only Newtonian methods tests
- `--hamiltonian` - Run only Hamiltonian methods tests

### Standalone Scripts (Alternative)

You can also run the scripts directly:

```bash
# Run tests directly
julia run_tests.jl                  # All tests
julia run_tests.jl --newtonian      # Newtonian only
julia run_tests.jl --hamiltonian    # Hamiltonian only

# Run trajectory optimization directly
julia run_trajectory_optimization.jl
```

## Project Structure

```
GRAVITy/
├── GRAVITy.jl                    # Main executable (use this!)
├── run_tests.jl                  # Test runner functions
├── run_trajectory_optimization.jl # Trajectory optimization functions
├── src/                          # Source modules
│   ├── common/
│   │   └── CommonUtils.jl        # Common utilities and functions
│   ├── newtonian/
│   │   └── NewtonianMethods.jl   # Classical numerical methods
│   ├── hamiltonian/
│   │   └── HamiltonianMethods.jl # Symplectic integrators
│   └── trajectory_optimization/
│       └── TrajectoryOptimization.jl # Trajectory optimization
├── data/                         # Input data (CSV files)
├── results/                      # Output results and plots
└── Project.toml                  # Dependencies
```
