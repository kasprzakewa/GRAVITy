module CommonUtils

using LinearAlgebra
using Statistics
using Printf
using CSV
using DataFrames
using Plots

export μ, r1_r2, u_eff, energy_integral_2d, energy_integral_hamiltonian
export newtonian_to_hamiltonian, hamiltonian_to_newtonian
export find_lagrange_points, get_test_cases
export create_plot, TestResult, save_results_to_csv, save_results_to_txt
export benchmark_memory, generate_methods_summary

const μ = 0.01215058560962404

function r1_r2(x, y, μ=μ)
    r1 = sqrt((x + μ)^2 + y^2)
    r2 = sqrt((x - 1 + μ)^2 + y^2)
    return r1, r2
end

function u_eff(x, y, μ=μ)
    r1, r2 = r1_r2(x, y, μ)
    U_eff = -(1 - μ)/r1 - μ/r2 - 0.5*(x^2 + y^2)
    return U_eff
end

function energy_integral_2d(x, y, vx, vy, μ=μ)
    potential = u_eff(x, y, μ)
    kinetic = 0.5 * (vx^2 + vy^2)
    return kinetic + potential
end

function energy_integral_hamiltonian(q, p, μ=μ)
    x, y = q
    px, py = p
    vx = px + y
    vy = py - x
    return energy_integral_2d(x, y, vx, vy, μ)
end

function newtonian_to_hamiltonian(u)
    x, y, vx, vy = u
    px = vx - y
    py = vy + x
    return ([x, y], [px, py])
end

function hamiltonian_to_newtonian(q, p)
    x, y = q
    px, py = p
    vx = px + y
    vy = py - x
    return [x, y, vx, vy]
end

function find_lagrange_points(μ=μ)
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

function get_test_cases()
    test_cases = []
    T_end_values = [
        10.0, 
        # 50.0, 
        # 100.0
    ]

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

    # for T_end in T_end_values
    #     push!(test_cases, Dict(
    #         "name" => "E_L1 < E < E_L2",
    #         "x0" => 0.8,
    #         "y0" => 0.0,
    #         "vx0" => 0.0,
    #         "vy0" => 0.15,
    #         "description" => "Quasi-periodic, E_L1 < E < E_L2",
    #         "T_end" => T_end
    #     ))
    # end

    # for T_end in T_end_values
    #     push!(test_cases, Dict(
    #         "name" => "E_L4 < E",
    #         "x0" => 0.5078494144,
    #         "y0" => 0.8560254038,
    #         "vx0" => -0.05,         
    #         "vy0" => 0.3,
    #         "description" => "High energy trajectory exploring L4 region, E > E_L4",
    #         "T_end" => T_end
    #     ))
    # end

    return test_cases
end

function create_plot(x_traj, y_traj, t_vals, energy_drift, method_name, method_case)
    p1 = plot(x_traj, y_traj, 
        label="Trajectory",
        xlabel="x",
        ylabel="y",
        color=:maroon4,
        title="Trajectory",
        aspect_ratio=:equal,
    )
    
    scatter!(p1, [-μ], [0], 
        color=:mediumblue, 
        markersize=8, 
        markershape=:circle,
        label="Earth")
    scatter!(p1, [1-μ], [0], 
        color=:burlywood2, 
        markersize=6, 
        markershape=:circle,
        label="Moon")
    
    lagrange_pts = find_lagrange_points()
    for (name, pos) in lagrange_pts
        if name in ["L1", "L2", "L3", "L4", "L5"]
            scatter!(p1, [pos[1]], [pos[2]], 
                color=:gold, 
                markersize=6, 
                markershape=:star,
                label=(name == "L1" ? "Lagrange points" : ""))
        end
    end
    
    p2 = plot(t_vals, energy_drift,
        label="Energy drift",
        xlabel="Time",
        ylabel="ΔE",
        color=:maroon4,
        title="Energy conservation",
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

function generate_methods_summary(results::Vector{TestResult}, method_type::String)
    println("\n" * "="^80)
    println("$(uppercase(method_type)) METHODS SUMMARY")
    println("="^80)
    
    method_groups = Dict{String, Vector{TestResult}}()
    
    for result in results
        base_method = result.method_name
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
            println("$i. $(result.method_name) ($(result.case_name)): |ΔE| = $(result.max_energy_drift)")
        end
    end
end

end
