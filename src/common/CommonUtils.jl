"""
CommonUtils Module

Wspólne narzędzia i funkcje dla analizy CR3BP (Circular Restricted Three-Body Problem).
Zawiera podstawowe funkcje matematyczne, konwersje między interpretacjami newtonowską 
i hamiltonowską, punkty Lagrange'a oraz zarządzanie wynikami testów.
"""

module CommonUtils

using LinearAlgebra
using Statistics
using Printf

# Try to import CSV and DataFrames, fallback if not available
try
    using CSV
    global csv_available = true
catch
    global csv_available = false
    println("Warning: CSV.jl not available - results will not be saved to CSV")
end

try
    using DataFrames
    global dataframes_available = true
catch
    global dataframes_available = false
    println("Warning: DataFrames.jl not available - results will not be saved to CSV")
end

# Try to import Plots, fallback if not available
try
    using Plots
    global plots_available = true
catch
    global plots_available = false
    println("Warning: Plots.jl not available - visualizations will be skipped")
end

# Export all public functions and constants
export μ, r1_r2, u_eff, energy_integral_2d, energy_integral_hamiltonian
export newtonian_to_hamiltonian, hamiltonian_to_newtonian
export find_lagrange_points, get_test_cases
export create_plot, TestResult, save_results_to_csv, save_results_to_txt
export benchmark_memory

const μ = 0.01215058560962404  # Earth-Moon mass ratio

"""
    r1_r2(x, y, μ=μ)

Oblicza odległości od punktu (x, y) do obu ciał w układzie CR3BP.

# Argumenty
- `x`: współrzędna x w układzie wirującym
- `y`: współrzędna y w układzie wirującym
- `μ`: stosunek mas (domyślnie dla układu Ziemia-Księżyc)

# Zwraca
- `r1`: odległość od pierwszego ciała (Ziemia)
- `r2`: odległość od drugiego ciała (Księżyc)
"""
function r1_r2(x, y, μ=μ)
    r1 = sqrt((x + μ)^2 + y^2)
    r2 = sqrt((x - 1 + μ)^2 + y^2)
    return r1, r2
end

"""
    u_eff(x, y, μ=μ)

Oblicza efektywny potencjał w układzie wirującym CR3BP.

# Argumenty
- `x`, `y`: współrzędne w układzie wirującym
- `μ`: stosunek mas

# Zwraca
Wartość efektywnego potencjału U_eff
"""
function u_eff(x, y, μ=μ)
    r1, r2 = r1_r2(x, y, μ)
    U_eff = -(1 - μ)/r1 - μ/r2 - 0.5*(x^2 + y^2)
    return U_eff
end

"""
    energy_integral_2d(x, y, vx, vy, μ=μ)

Oblicza całkę energii (stałą Jacobiego) w interpretacji newtonowskiej.

# Argumenty
- `x`, `y`: współrzędne położenia
- `vx`, `vy`: prędkości
- `μ`: stosunek mas

# Zwraca
Wartość całki energii E = T + U
"""
function energy_integral_2d(x, y, vx, vy, μ=μ)
    potential = u_eff(x, y, μ)
    kinetic = 0.5 * (vx^2 + vy^2)
    return kinetic + potential
end

"""
    energy_integral_hamiltonian(q, p, μ=μ)

Oblicza całkę energii w interpretacji hamiltonowskiej.

# Argumenty
- `q`: wektor współrzędnych uogólnionych [x, y]
- `p`: wektor pędów uogólnionych [px, py]
- `μ`: stosunek mas

# Zwraca
Wartość energii hamiltonowskiej
"""
function energy_integral_hamiltonian(q, p, μ=μ)
    x, y = q
    px, py = p
    vx = px + y
    vy = py - x
    return energy_integral_2d(x, y, vx, vy, μ)
end

"""
    newtonian_to_hamiltonian(u)

Konwertuje zmienne newtonowskie do hamiltonowskich.

# Argumenty
- `u`: wektor [x, y, vx, vy] w interpretacji newtonowskiej

# Zwraca
- `q`: współrzędne uogólnione [x, y]
- `p`: pędy uogólnione [px, py]
"""
function newtonian_to_hamiltonian(u)
    x, y, vx, vy = u
    px = vx - y
    py = vy + x
    return ([x, y], [px, py])
end

"""
    hamiltonian_to_newtonian(q, p)

Konwertuje zmienne hamiltonowskie do newtonowskich.

# Argumenty
- `q`: współrzędne uogólnione [x, y]
- `p`: pędy uogólnione [px, py]

# Zwraca
Wektor [x, y, vx, vy] w interpretacji newtonowskiej
"""
function hamiltonian_to_newtonian(q, p)
    x, y = q
    px, py = p
    vx = px + y
    vy = py - x
    return [x, y, vx, vy]
end

"""
    find_lagrange_points(μ=μ)

Wyznacza przybliżone położenia pięciu punktów Lagrange'a.

# Argumenty
- `μ`: stosunek mas

# Zwraca
Słownik z pozycjami L1, L2, L3, L4, L5
"""
function find_lagrange_points(μ=μ)
    # Approximate Lagrange points for CR3BP
    l1_x = 1 - (μ/3)^(1/3)
    l2_x = 1 + (μ/3)^(1/3)
    l3_x = -1 - 5/12*μ
    l4_x = 0.5 - μ
    l4_y = sqrt(3)/2
    l5_x = 0.5 - μ
    l5_y = -sqrt(3)/2
    
    return Dict(
        "L1" => [l1_x, 0.0],
        "L2" => [l2_x, 0.0],
        "L3" => [l3_x, 0.0],
        "L4" => [l4_x, l4_y],
        "L5" => [l5_x, l5_y]
    )
end

"""
    get_test_cases()

Generuje zestaw przypadków testowych dla różnych poziomów energii w CR3BP.

# Zwraca
Wektor słowników z warunkami początkowymi i parametrami testów
"""
function get_test_cases()
    lagrange_points = find_lagrange_points()
    
    E_L1 = energy_integral_2d(lagrange_points["L1"][1], lagrange_points["L1"][2], 0.0, 0.0)
    E_L2 = energy_integral_2d(lagrange_points["L2"][1], lagrange_points["L2"][2], 0.0, 0.0)
    E_L4 = energy_integral_2d(lagrange_points["L4"][1], lagrange_points["L4"][2], 0.0, 0.0)
    
    test_cases = []
    T_end_values = [10.0, 50.0, 100.0]

    for T_end in T_end_values
        push!(test_cases, Dict(
            "name" => "E < E_L1",
            "x0" => 0.8,
            "y0" => 0.0,
            "vx0" => 0.0,
            "vy0" => 0.05,
            "description" => "Bound orbit, E < E_L1",
            "T_end" => T_end
        ))
    end

    for T_end in T_end_values
        push!(test_cases, Dict(
            "name" => "E_L1 < E < E_L2",
            "x0" => 0.8,
            "y0" => 0.0,
            "vx0" => 0.0,
            "vy0" => 0.15,
            "description" => "Quasi-periodic, E_L1 < E < E_L2",
            "T_end" => T_end
        ))
    end

    for T_end in T_end_values
        push!(test_cases, Dict(
            "name" => "E_L4 < E",
            "x0" => 0.5078494144,
            "y0" => 0.8560254038,
            "vx0" => -0.05,         
            "vy0" => 0.3,
            "description" => "High energy trajectory exploring L4 region, E > E_L4",
            "T_end" => T_end
        ))
    end

    return test_cases
end

function create_plot(x_traj, y_traj, t_vals, energy_drift, method_name, method_case)
    if !plots_available
        println("Plots.jl not available - skipping visualization for $method_name")
        return nothing
    end
    
    p1 = plot(x_traj, y_traj, 
        label="Trajektoria",
        xlabel="x",
        ylabel="y",
        title="$method_name",
        aspect_ratio=:equal,
    )
    
    scatter!(p1, [-μ], [0], 
        color=:limegreen, 
        markersize=8, 
        markershape=:circle,
        label="Ziemia")
    scatter!(p1, [1-μ], [0], 
        color=:magenta, 
        markersize=6, 
        markershape=:circle,
        label="Księżyc")
    
    lagrange_pts = find_lagrange_points()
    for (name, pos) in lagrange_pts
        if name in ["L1", "L2", "L3", "L4", "L5"]
            scatter!(p1, [pos[1]], [pos[2]], 
                color=:red, 
                markersize=4, 
                markershape=:star,
                label=(name == "L1" ? "Punkty Lagrange'a" : ""))
        end
    end
    
    p2 = plot(t_vals, energy_drift,
        label="Odchylenie energii",
        xlabel="Czas",
        ylabel="ΔE",
        title="Zachowanie energii",
    )
    
    combined = plot(p1, p2, layout=(2,1), size=(900, 800))
    return combined
end

mutable struct TestResult
    method_name::String
    case_name::String
    dt::Float64
    T::Float64
    max_energy_drift::Float64
    execution_time::Float64
    memory_usage::Float64
    final_time::Float64
    mean_energy_drift::Float64
    std_energy_drift::Float64

    function TestResult(method_name, case_name, dt=0.0, T=0.0)
        new(method_name, case_name, dt, T, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function save_results_to_csv(results::Vector{TestResult}, filename::String)
    if !csv_available || !dataframes_available
        println("CSV/DataFrames packages not available - saving results to text file instead")
        save_results_to_txt(results, replace(filename, ".csv" => ".txt"))
        return
    end
    
    df = DataFrame(
        method=String[],
        test_case=String[],
        dt=Float64[],
        T=Float64[],
        max_energy_drift=Float64[],
        execution_time_s=Float64[],
        memory_mb=Float64[],
        final_time=Float64[],
        mean_energy_drift=Float64[],
        std_energy_drift=Float64[]
    )

    for result in results
        push!(df, (
            result.method_name,
            result.case_name,
            result.dt,
            result.T,
            result.max_energy_drift,
            result.execution_time,
            result.memory_usage,
            result.final_time,
            result.mean_energy_drift,
            result.std_energy_drift
        ))
    end

    CSV.write(filename, df)
    println("Results saved to $filename")
end

function save_results_to_txt(results::Vector{TestResult}, filename::String)
    open(filename, "w") do f
        write(f, "# CR3BP Numerical Methods Test Results\n")
        write(f, "# method,test_case,dt,T,max_energy_drift,execution_time_s,memory_mb,final_time,mean_energy_drift,std_energy_drift\n")

        for result in results
            write(f, "$(result.method_name),$(result.case_name),$(result.dt),$(result.T),$(result.max_energy_drift),$(result.execution_time),$(result.memory_usage),$(result.final_time),$(result.mean_energy_drift),$(result.std_energy_drift)\n")
        end
    end
    println("Results saved to $filename")
end

function benchmark_memory()
    GC.gc()
    return Base.gc_bytes()
end

end
