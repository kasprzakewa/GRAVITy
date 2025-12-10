"""Newtonian formulation and standard integrators for PCR3BP.

Implements classical integration methods including Euler, Runge-Kutta,
and Adams-Bashforth.
"""
module NewtonianMethods

using DifferentialEquations: ODEProblem, solve, Euler, Midpoint, RK4, Vern9, DP8, AB5
using StaticArrays
using Printf
using Statistics
using LinearAlgebra
using Plots

import ..CommonUtils: μ, energy_integral_2d
import ..CommonUtils: TestResult, create_plot, save_results_to_csv, benchmark_memory
import ..CommonUtils: get_test_cases, generate_methods_summary

export cr3bp_newtonian!
export test_newtonian_method, run_newtonian_tests

"""Equations of motion for PCR3BP in newtonian formulation."""
function cr3bp_newtonian!(du, u, p, t)
    μ = p
    x, y, vx, vy = u
    
    r1² = (x + μ)^2 + y^2
    r2² = (x - (1 - μ))^2 + y^2
    r1³ = r1²^(3/2)
    r2³ = r2²^(3/2)
    
    ax_grav = -(1 - μ) * (x + μ) / r1³ - μ * (x - (1 - μ)) / r2³
    ay_grav = -(1 - μ) * y / r1³ - μ * y / r2³
    
    du[1] = vx                           
    du[2] = vy                           
    du[3] = 2*vy + x + ax_grav          
    du[4] = -2*vx + y + ay_grav         
end

"""Create ODE problem for PCR3BP in newtonian form."""
function create_newtonian_problem(u0, tspan)
    return ODEProblem(cr3bp_newtonian!, u0, tspan, μ)
end

"""Create ODE problem with custom dynamics function."""
function create_newtonian_problem_with_function(u0, tspan, f)
    return ODEProblem(f, u0, tspan, μ)
end

"""Solve newtonian ODE problem with specified method and timestep."""
function solve_newtonian_problem(prob, method; dt=0.01)
    return solve(prob, method, dt=dt, adaptive=false)
end

"""Test single newtonian method on given test case."""
function test_newtonian_method(method, method_name, test_case; dt_values=[0.01, 0.001, 0.0001])
    results = TestResult[]
    
    for dt in dt_values
        x0, y0 = test_case["x0"], test_case["y0"]
        vx0, vy0 = test_case["vx0"], test_case["vy0"]
        u0 = [x0, y0, vx0, vy0]
        tspan = (0.0, test_case["T_end"])
        
        prob = create_newtonian_problem(u0, tspan)
        result = TestResult(method_name, test_case["name"], dt, test_case["T_end"])
        
        try
            mem_before = benchmark_memory()
            time_start = time()

            if method isa Symbol
                sol = solve_newtonian_problem(prob, eval(method)(); dt=dt)
            else
                sol = solve_newtonian_problem(prob, method; dt=dt)
            end
            
            time_end = time()
            mem_after = benchmark_memory()

            result.execution_time = time_end - time_start
            result.memory_usage = (mem_after - mem_before) / 1024^2
            result.final_time = sol.t[end]
            
            E_vals = [energy_integral_2d(sol[1,i], sol[2,i], sol[3,i], sol[4,i]) 
                     for i in 1:length(sol.t)]
            energy_drift = E_vals .- E_vals[1]
            
            result.max_energy_drift = maximum(abs.(energy_drift))
            result.mean_energy_drift = mean(abs.(energy_drift))
            result.std_energy_drift = std(energy_drift)
            
            x_traj = [sol[1,i] for i in 1:length(sol.t)]
            y_traj = [sol[2,i] for i in 1:length(sol.t)]
            
            plt = create_plot(x_traj, y_traj, sol.t, energy_drift, 
                            "$method_name (dt=$dt, T=$(test_case["T_end"]), $(test_case["name"]))", "")
            
            if plt !== nothing
                mkpath("results/newtonian_methods")
                filename = joinpath("results/newtonian_methods", 
                    "$(method_name)_$(test_case["name"])_dt_$(replace(string(dt), "." => "_")).png")
                try
                    savefig(plt, filename)
                catch e
                    println("    Warning: Could not save plot: $e")
                end
            end
            
            push!(results, result)
            
            @printf("    %-25s %-25s dt=%-8g T=%-8g Max|ΔE|=%-12.4e  Time=%.3fs\n", 
                method, test_case["name"], dt, test_case["T_end"], 
                result.max_energy_drift, result.execution_time)
            
        catch e
            println("    ERROR: $e")
            result.max_energy_drift = Inf
            push!(results, result)
        end
    end
    
    return results
end

"""Run all newtonian method tests and generate summary."""
function run_newtonian_tests()
    println("\n" * "="^40)
    println("NEWTONIAN METHODS")
    println("="^40)
    
    methods = [
        (:Euler, "Euler method"),
        (:Midpoint, "Midpoint method"),
        (:RK4, "Runge-Kutta 4th order method"),
        (:Vern9, "Verner's 9th order method"),
        (:DP8, "Dormand-Prince 8th order method"),
        (:AB5, "Adams-Bashforth 5th order method")
    ]
    
    test_cases = get_test_cases()
    all_results = TestResult[]
    
    for (method_symbol, method_name) in methods
        println("\n" * "-"^40)
        println("METHOD: $method_name")
        println("-"^40)
        
        for test_case in test_cases
            case_results = test_newtonian_method(method_symbol, method_name, test_case)
            append!(all_results, case_results)
        end
    end
    
    mkpath("results/newtonian_methods")
    save_results_to_csv(all_results, joinpath("results/newtonian_methods", "newtonian_methods_results.csv"))
    
    generate_methods_summary(all_results, "Newtonian")
    
    return all_results
end

end
