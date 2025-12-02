module NewtonianMethods

using DifferentialEquations: ODEProblem, solve, Euler, Midpoint, RK4, Vern9, DP8, AB5
using StaticArrays
using Printf
using Statistics
using LinearAlgebra
using Plots

import ..CommonUtils: μ, energy_integral_2d
import ..CommonUtils: TestResult, create_plot, save_results_to_csv, benchmark_memory
import ..CommonUtils: get_test_cases

export cr3bp_newtonian!
export test_newtonian_method, run_newtonian_tests, generate_newtonian_summary

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

function create_newtonian_problem(u0, tspan)
    return ODEProblem(cr3bp_newtonian!, u0, tspan, μ)
end

function create_newtonian_problem_with_function(u0, tspan, f)
    return ODEProblem(f, u0, tspan, μ)
end

function solve_newtonian_problem(prob, method; dt=0.01)
    return solve(prob, method, dt=dt, adaptive=false)
end

function test_newtonian_method(method, method_name, test_case; dt_values=[0.01, 0.001, 0.0001])
    results = TestResult[]
    
    println("Testing $method_name on case: $(test_case["name"])")
    
    for dt in dt_values
        println("  dt = $dt")
        
        x0, y0 = test_case["x0"], test_case["y0"]
        vx0, vy0 = test_case["vx0"], test_case["vy0"]
        u0 = [x0, y0, vx0, vy0]
        tspan = (0.0, test_case["T_end"])
        
        prob = create_newtonian_problem(u0, tspan)
        result = TestResult(method_name * "_dt_$dt", test_case["name"], dt, test_case["T_end"])
        
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
                mkpath("results/newtonian_output")
                filename = joinpath("results/newtonian_output", 
                    "cr3bp_newtonian_$(method_name)_$(test_case["name"])_dt_$(replace(string(dt), "." => "_")).png")
                try
                    savefig(plt, filename)
                catch e
                    println("    Warning: Could not save plot: $e")
                end
            end
            
            push!(results, result)
            
            println("    Max |ΔE| = $(result.max_energy_drift)")
            println("    Time = $(result.execution_time)s")
            
        catch e
            println("    ERROR: $e")
            result.max_energy_drift = Inf
            push!(results, result)
        end
    end
    
    return results
end

function run_newtonian_tests()
    println("="^80)
    println("NEWTONIAN METHODS TESTING FOR CR3BP")
    println("="^80)
    
    methods = [
        (:Euler, "Metoda Eulera"),
        (:Midpoint, "Metoda punktu środkowego"),
        (:RK4, "Metoda Rungego-Kutty 4. rzędu"),
        (:Vern9, "Metoda Vernera 9. rzędu"),
        (:DP8, "Metoda Dormanda-Prince'a 7. rzędu"),
        (:AB5, "Metoda Adamsa-Bashfortha 5. rzędu")
    ]
    
    test_cases = get_test_cases()
    all_results = TestResult[]
    
    for (method_symbol, method_name) in methods
        println("\n" * "="^60)
        println("METHOD: $method_name")
        println("="^60)
        
        for test_case in test_cases
            case_results = test_newtonian_method(method_symbol, method_name, test_case)
            append!(all_results, case_results)
        end
    end
    
    mkpath("results/newtonian_output")
    save_results_to_csv(all_results, joinpath("results/newtonian_output", "newtonian_methods_results.csv"))
    
    generate_newtonian_summary(all_results)
    
    return all_results
end

function generate_newtonian_summary(results::Vector{TestResult})
    println("\n" * "="^80)
    println("NEWTONIAN METHODS SUMMARY")
    println("="^80)
    
    method_groups = Dict{String, Vector{TestResult}}()
    
    for result in results
        base_method = split(result.method_name, "_dt_")[1]
        if !haskey(method_groups, base_method)
            method_groups[base_method] = TestResult[]
        end
        push!(method_groups[base_method], result)
    end
    
    for (method, method_results) in method_groups
        if isempty(method_results)
            continue
        end
        
        println("\n$method:")
        println("-"^40)
        
        for case_name in unique([r.case_name for r in method_results])
            case_results = filter(r -> r.case_name == case_name, method_results)
            if !isempty(case_results)
                best = argmin([r.max_energy_drift for r in case_results])
                best_result = case_results[best]
                println("  $case_name: Best |ΔE| = $(best_result.max_energy_drift) " *
                       "(dt=$(best_result.dt), time=$(best_result.execution_time)s)")
            end
        end
    end
    
    println("\n" * "="^40)
    println("OVERALL RANKING BY ENERGY CONSERVATION")
    println("="^40)
    
    best_results = TestResult[]
    for (method, method_results) in method_groups
        for case_name in unique([r.case_name for r in method_results])
            case_results = filter(r -> r.case_name == case_name, method_results)
            if !isempty(case_results) && all(isfinite(r.max_energy_drift) for r in case_results)
                best_idx = argmin([r.max_energy_drift for r in case_results])
                push!(best_results, case_results[best_idx])
            end
        end
    end

    if !isempty(best_results)
        sorted_results = sort(best_results, by=r -> r.max_energy_drift)
        
        for (i, result) in enumerate(sorted_results[1:min(10, length(sorted_results))])
            method_clean = split(result.method_name, "_dt_")[1]
            println("$i. $method_clean ($(result.case_name)): |ΔE| = $(result.max_energy_drift)")
        end
    end
end

end
