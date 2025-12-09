# GRAVITy Installation Guide

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/kasprzakewa/GRAVITy.git
cd GRAVITy
```

### 2. Install all dependencies automatically
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This will automatically install all required packages listed in `Project.toml` and `Manifest.toml`.

### 3. Run the project
```bash
julia --project=. GRAVITy.jl help
```

## Required Packages

The following packages will be automatically installed:
- **CSV** - Reading/writing CSV data files
- **DataFrames** - Tabular data manipulation and analysis
- **DifferentialEquations** - Comprehensive ODE/DAE solver suite
- **GeometricIntegrators** - Symplectic and variational integrators (Gauss, Lobatto, EPRK)
- **NearestNeighbors** - KD-tree implementation for trajectory optimization
- **Plots** - Plotting and visualization framework
- **RungeKutta** - Additional Runge-Kutta method implementations
- **SimpleSolvers** - Simple numerical solvers for nonlinear systems
- **StaticArrays** - High-performance fixed-size arrays

## Troubleshooting

### If packages fail to install:
1. Update Julia package registry:
```bash
julia --project=. -e 'using Pkg; Pkg.update()'
```

2. Try reinstalling:
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

### If you encounter version conflicts:
The `Manifest.toml` file locks all package versions to ensure compatibility. If you need to update packages, run:
```bash
julia --project=. -e 'using Pkg; Pkg.update()'
```

## System Requirements

- Julia 1.10.0 or higher (tested with Julia 1.10.0)
