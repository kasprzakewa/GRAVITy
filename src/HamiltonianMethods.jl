"""Hamiltonian formulation and symplectic integrators for PCR3BP.

Implements various symplectic methods including Gauss, Lobatto, and explicit
partitioned Runge-Kutta methods.
"""
module HamiltonianMethods

using GeometricIntegrators: GeometricIntegrator, integrate, HODEProblem, 
                        ImplicitMidpoint, Gauss,
                        LobattoIIIC, LobattoIIIAIIIB, LobattoIIIBIIIA, 
                        EPRK
using SimpleSolvers
using Printf
using RungeKutta
using Statistics
using Plots

import ..CommonUtils: μ, r1_r2, u_eff, energy_integral_hamiltonian
import ..CommonUtils: TestResult, create_plot, save_results_to_csv, benchmark_memory
import ..CommonUtils: get_test_cases, generate_methods_summary

export grad_p_H, grad_q_H, vfield, ffield, hamiltonian_function
export create_hamiltonian_problem, integrate_hamiltonian_method
export test_hamiltonian_method
export run_hamiltonian_tests

"""Compute gradient of hamiltonian with respect to momentum p."""
function grad_p_H(q, p)
    x, y = q
    px, py = p
    return [px + y, py - x]
end

"""Compute gradient of hamiltonian with respect to position q."""
function grad_q_H(q, p; μ=μ)
    x, y = q
    px, py = p

    r1, r2 = r1_r2(x, y, μ)

    dUx = (1 - μ)*(x + μ)/r1^3 + μ*(x - 1 + μ)/r2^3 - x
    dUy = (1 - μ)*y/r1^3 + μ*y/r2^3 - y

    return [py - x - dUx, -px - y - dUy]
end

"""Velocity field for hamiltonian system."""
function vfield(v, t, q, p, params)
    v[1], v[2] = grad_p_H(q, p)
end

"""Force field for hamiltonian system."""
function ffield(f, t, q, p, params)
    μ = params.μ
    f[1], f[2] = grad_q_H(q, p; μ=μ)
end

"""Hamiltonian of the system."""
function hamiltonian_function(t, q, p, params)
    x, y = q
    px, py = p
    μ = params.μ
    vx, vy = grad_p_H(q, p)

    T = 0.5 * (vx^2 + vy^2)
    U = u_eff(x, y, μ)
    
    return T + U
end

"""Create hamiltonian ODE problem for integration."""
function create_hamiltonian_problem(q0, p0, tspan, dt; μ=μ)
    params = (μ=μ,)
    prob = HODEProblem(vfield, ffield, hamiltonian_function, tspan, dt, q0, p0; parameters=params)
    return prob
end

"""Select and configure integrator for specified method."""
function get_integrator(prob, method_function::String)
    newton_solver = SimpleSolvers.Newton()
    
    integrators = Dict(
        "ImplicitMidpoint" => ImplicitMidpoint(),
        "Gauss(2)" => Gauss(2),
        "Gauss(3)" => Gauss(3),
        "LobattoIIIC(3)" => LobattoIIIC(3),
        "LobattoIIIC(4)" => LobattoIIIC(4),
        "LobattoIIIAIIIB(3)" => LobattoIIIAIIIB(3),
        "LobattoIIIAIIIB(4)" => LobattoIIIAIIIB(4),
        "LobattoIIIBIIIA(3)" => LobattoIIIBIIIA(3),
        "LobattoIIIBIIIA(4)" => LobattoIIIBIIIA(4),
        "ERK2" => EPRK(RungeKutta.PartitionedTableau(
            RungeKutta.Tableaus.TableauHeun2(),
            RungeKutta.Tableaus.TableauHeun2()
        ))
    )
    
    methods_with_solver = [
        "LobattoIIIC(3)", "LobattoIIIC(4)",
        "LobattoIIIAIIIB(3)", "LobattoIIIAIIIB(4)",
        "LobattoIIIBIIIA(3)", "LobattoIIIBIIIA(4)"
    ]
    
    if !haskey(integrators, method_function)
        @warn "Unknown method: $method_function, using Gauss(2) as default"
        method = Gauss(2)
        return GeometricIntegrator(prob, method)
    end
    
    method = integrators[method_function]
    
    if method_function in methods_with_solver
        return GeometricIntegrator(prob, method, solver=newton_solver)
    else
        return GeometricIntegrator(prob, method)
    end
end

"""Integrate hamiltonian problem using specified method."""
function integrate_hamiltonian_method(prob, method_function::String)
    integrator = get_integrator(prob, method_function)
    sol = integrate(integrator)
    return sol
end

"""Test single hamiltonian method on given test case."""
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
            method_name, 
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
                mkpath("results/hamiltonian_methods")
                filename = joinpath("results/hamiltonian_methods", 
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

"""Run all hamiltonian method tests and generate summary."""
function run_hamiltonian_tests()
    separable_methods = [
        ("ERK2", "Explicit Runge-Kutta method with Heun2 tableau")
    ]
    
    nonseparable_methods = [
        ("ImplicitMidpoint", "Implicit midpoint"),
        ("Gauss(2)", "Gauss 4th order method"),
        ("Gauss(3)", "Gauss 6th order method"),
        ("LobattoIIIC(3)", "LobattoIIIC 4th order method"),
        ("LobattoIIIC(4)", "LobattoIIIC 6th order method"),
        ("LobattoIIIAIIIB(3)", "LobattoIIIAIIIB 4th order method"),
        ("LobattoIIIAIIIB(4)", "LobattoIIIAIIIB 6th order method"),
        ("LobattoIIIBIIIA(3)", "LobattoIIIBIIIA 4th order method"),
        ("LobattoIIIBIIIA(4)", "LobattoIIIBIIIA 6th order method")
    ]
    
    test_cases = get_test_cases()
    all_results = TestResult[]

    if !isempty(separable_methods)
        println("\n" * "="^40)
        println("SEPARABLE HAMILTONIAN METHODS")
        println("="^40)
        
        for (method_function, method_name) in separable_methods
            println("\n" * "-"^40)
            println("METHOD: $method_name")
            println("-"^40)
            
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
        println("\n" * "-"^40)
        println("METHOD: $method_name")
        println("-"^40)
        
        for test_case in test_cases
            case_results = test_hamiltonian_method(method_function, method_name, test_case)
            append!(all_results, case_results)
        end
    end
    
    mkpath("results/hamiltonian_methods")
    save_results_to_csv(all_results, joinpath("results/hamiltonian_methods", "hamiltonian_methods_results.csv"))
  
    generate_methods_summary(all_results, "Hamiltonian")
    
    return all_results
end

end
