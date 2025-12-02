"""
HamiltonianMethods Module

Metody numeryczne oparte na interpretacji hamiltonowskiej CR3BP.
Używa symplektycznych integratorów z GeometricIntegrators.jl do zachowania
struktury hamiltonowskiej układu dynamicznego.
"""

module HamiltonianMethods

using GeometricIntegrators: GeometricIntegrator, integrate, HODEProblem, 
                          SymplecticEulerA, SymplecticEulerB, ImplicitMidpoint, Gauss,
                          LobattoIIIC, LobattoIIIAIIIB, LobattoIIIBIIIA, EPRK
using StaticArrays
using SimpleSolvers
using Printf
using RungeKutta
using Statistics
using LinearAlgebra
using Plots

import ..CommonUtils: μ, r1_r2, u_eff, energy_integral_hamiltonian
import ..CommonUtils: TestResult, create_plot, save_results_to_csv, benchmark_memory
import ..CommonUtils: get_test_cases

export grad_p_H, grad_q_H, vfield, ffield, hamiltonian_function
export create_hamiltonian_problem, integrate_hamiltonian_method
export plot_hamiltonian_solution, test_hamiltonian_method
export run_hamiltonian_tests, generate_hamiltonian_summary
export run_separate_case_hamiltonian_test

function grad_p_H(q, p)
    x, y = q
    px, py = p
    return [px + y, py - x]
end

function grad_q_H(q, p; μ=μ)
    x, y = q
    px, py = p

    r1, r2 = r1_r2(x, y, μ)

    dUx = (1 - μ)*(x + μ)/r1^3 + μ*(x - 1 + μ)/r2^3 - x
    dUy = (1 - μ)*y/r1^3 + μ*y/r2^3 - y

    return [py - x - dUx, -px - y - dUy]
end

function vfield(v, t, q, p, params)
    v[1], v[2] = grad_p_H(q, p)
end

function ffield(f, t, q, p, params)
    μ = params.μ
    f[1], f[2] = grad_q_H(q, p; μ=μ)
end

function hamiltonian_function(t, q, p, params)
    x, y = q
    px, py = p
    μ = params.μ
    vx, vy = grad_p_H(q, p)

    T = 0.5 * (vx^2 + vy^2)
    U = u_eff(x, y, μ)
    
    return T + U
end

function create_hamiltonian_problem(q0, p0, tspan, dt; μ=μ)
    params = (μ=μ,)
    prob = HODEProblem(vfield, ffield, hamiltonian_function, tspan, dt, q0, p0; parameters=params)
    return prob
end

function integrate_hamiltonian_method(prob, method_function::String)
    if method_function == "Implicit_Midpoint"
        integrator = GeometricIntegrator(prob, ImplicitMidpoint())
    elseif method_function == "Gauss_Order_4"
        integrator = GeometricIntegrator(prob, Gauss(2))
    elseif method_function == "Gauss_Order_6"
        integrator = GeometricIntegrator(prob, Gauss(3))
    elseif method_function == "LobattoIIIC_Order_4"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIC(3),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "LobattoIIIC_Order_6"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIC(4),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "LobattoIIIAIIIB_Order_4"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIAIIIB(3),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "LobattoIIIAIIIB_Order_6"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIAIIIB(4),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "LobattoIIIBIIIA_Order_4"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIBIIIA(3),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "LobattoIIIBIIIA_Order_6"
        integrator = GeometricIntegrator(
            prob,
            LobattoIIIBIIIA(4),
            solver = SimpleSolvers.Newton()
        )
    elseif method_function == "ERK2"
        tableau = RungeKutta.Tableaus.TableauHeun2()
        PRK = RungeKutta.PartitionedTableau(tableau, tableau)
        integrator = GeometricIntegrator(prob, EPRK(PRK))
    else
        println("Unknown method: $method_function, using Implicit Midpoint as default")
        integrator = GeometricIntegrator(prob, Gauss(2))
    end
    sol = integrate(integrator)
    return sol
end

function plot_hamiltonian_solution(plt, sol; color=:black, alpha=1.0, label=false)
    xs = [sol.s.q.d[i][1] for i in eachindex(sol.s.q.d)]
    ys = [sol.s.q.d[i][2] for i in eachindex(sol.s.q.d)]
    
    plot!(plt, xs, ys, label=label, color=color, alpha=alpha)
    return plt
end

function test_hamiltonian_method(method_function, method_name, test_case; dt_values=[0.01, 0.001, 0.0001])
    results = TestResult[]
    
    for dt in dt_values
        x0, y0 = test_case["x0"], test_case["y0"]
        vx0, vy0 = test_case["vx0"], test_case["vy0"]
        
        px0 = vx0 - y0
        py0 = vy0 + x0
        
        q0 = [x0, y0]
        p0 = [px0, py0]
        
        tspan = (0.0, test_case["T_end"])
        
        result = TestResult(
            method_function, 
            test_case["name"], 
            dt, 
            test_case["T_end"]
        )
        
        try
            prob = create_hamiltonian_problem(q0, p0, tspan, dt; μ=μ)
            
            mem_before = benchmark_memory()
            time_start = time()
            
            sol = integrate_hamiltonian_method(prob, method_function)
            
            time_end = time()
            mem_after = benchmark_memory()
            
            result.execution_time = time_end - time_start
            result.memory_usage = (mem_after - mem_before) / 1024^2
            result.final_time = sol.t[end]

            q_series = sol.s.q
            p_series = sol.s.p

            q_vals = [collect(q_series[i]) for i in eachindex(q_series)]
            p_vals = [collect(p_series[i]) for i in eachindex(p_series)]

            E_vals = [
                energy_integral_hamiltonian(q_vals[i], p_vals[i])
                for i in eachindex(q_vals)
            ]

            energy_drift = E_vals .- first(E_vals)
            
            result.max_energy_drift = maximum(abs.(energy_drift))
            result.mean_energy_drift = mean(abs.(energy_drift))
            result.std_energy_drift = std(energy_drift)
            
            xs = [q_vals[i][1] for i in eachindex(q_vals)]
            ys = [q_vals[i][2] for i in eachindex(q_vals)]
            ts = sol.t

            plt = create_plot(xs, ys, ts, energy_drift, 
                            "$method_name (dt=$dt, T=$(test_case["T_end"]), $(test_case["name"]))", "")

            if plt !== nothing
                mkpath("results/hamiltonian_output")
                filename = joinpath("results/hamiltonian_output", 
                    "$(method_function)_$(test_case["name"])_T=$(test_case["T_end"])_dt_$(replace(string(dt), "." => "_")).png")
                try
                    savefig(plt, filename)
                catch e
                    println("    Warning: Could not save plot: $e")
                end
            end
            
            push!(results, result)
            
            @printf("    %-25s %-25s dt=%-8g T=%-8g Max|ΔE|=%-12.4e  Time=%.3fs\n", 
                method_function, test_case["name"], dt, test_case["T_end"], 
                result.max_energy_drift, result.execution_time)
            
        catch e
            println("    ERROR: $e")
            result.max_energy_drift = Inf
            push!(results, result)
        end
    end
    
    return results
end

function run_hamiltonian_tests()
    separable_methods = [
        ("ERK2", "Jawna metoda RK rzędu drugiego z tablicą Heuna")
    ]
    
    nonseparable_methods = [
        ("Implicit_Midpoint", "Niejawna metoda punktu środkowego"),
        ("Gauss_Order_4", "Metoda Gaussa rzędu czwartego"),
        ("Gauss_Order_6", "Metoda Gaussa rzędu szóstego"),
        ("LobattoIIIC_Order_4", "Metoda LobattoIIIC rzędu czwartego"),
        ("LobattoIIIC_Order_6", "Metoda LobattoIIIC rzędu szóstego"),
        ("LobattoIIIAIIIB_Order_4", "Metoda LobattoIIIAIIIB rzędu czwartego"),
        ("LobattoIIIAIIIB_Order_6", "Metoda LobattoIIIAIIIB rzędu szóstego"),
        ("LobattoIIIBIIIA_Order_4", "Metoda LobattoIIIBIIIA rzędu czwartego"),
        ("LobattoIIIBIIIA_Order_6", "Metoda LobattoIIIBIIIA rzędu szóstego")
    ]
    
    test_cases = get_test_cases()
    all_results = TestResult[]

    if !isempty(separable_methods)
        println("\n" * "="^40)
        println("SEPARABLE HAMILTONIAN METHODS")
        println("="^40)
        
        for (method_function, method_name) in separable_methods
            println("\n" * "-"^20)
            println("METHOD: $method_name")
            println("-"^20)
            
            for test_case in test_cases
                case_results = test_hamiltonian_method(method_function, method_name, test_case)
                append!(all_results, case_results)
            end
        end
    end
    
    println("\n" * "="^40)
    println("NON-SEPARABLE HAMILTONIAN METHODS")
    println("="^40)
    
    for (method_function, method_name) in nonseparable_methods
        println("\n" * "-"^20)
        println("METHOD: $method_name")
        println("-"^20)
        
        for test_case in test_cases
            case_results = test_hamiltonian_method(method_function, method_name, test_case)
            append!(all_results, case_results)
        end
    end
    
    mkpath("results/hamiltonian_output")
    save_results_to_csv(all_results, joinpath("results/hamiltonian_output", "hamiltonian_methods_results.csv"))
  
    generate_hamiltonian_summary(all_results)
    
    return all_results
end

function generate_hamiltonian_summary(results::Vector{TestResult})
    println("\n" * "="^80)
    println("HAMILTONIAN METHODS SUMMARY")
    println("="^80)
    
    method_groups = Dict{String, Vector{TestResult}}()
    
    for result in results
        base_method = split(result.method_name, "_dt_")[1]
        if !haskey(method_groups, base_method)
            method_groups[base_method] = TestResult[]
        end
        push!(method_groups[base_method], result)
    end
    
    for (method, method_results) in sort(collect(method_groups), by=x->x[1])
        if isempty(method_results)
            continue
        end
        
        println("\n$method:")
        println("-"^40)
        
        for case_name in unique([r.case_name for r in method_results])
            case_results = filter(r -> r.case_name == case_name, method_results)
            if !isempty(case_results)
                valid_results = filter(r -> isfinite(r.max_energy_drift), case_results)
                if !isempty(valid_results)
                    best = argmin([r.max_energy_drift for r in valid_results])
                    best_result = valid_results[best]
                    println("  $case_name: Best |ΔE| = $(best_result.max_energy_drift) " *
                           "(dt=$(best_result.dt), time=$(best_result.execution_time)s)")
                end
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

function run_separate_case_hamiltonian_test()
    println("\n" * "="^40)
    println("SINGLE CASE HAMILTONIAN METHOD TEST")
    println("="^40)

    test_case = Dict(
        "name" => "Lagrange_L1_Orbit",
        "x0" => 0.27296264470902415,
        "y0" => -0.0630518166674764,
        "vx0" => -0.08604980548541422,
        "vy0" => 1.9206667385409442,
        "T_end" => 10.0
    )

    result = test_hamiltonian_method(
        "LobattoIIIAIIIB_Order_6", 
        "Metoda LobattoIIIAIIIB rzędu szóstego", 
        test_case; 
        dt_values=[0.001]
    )
    
    return result
end

end
