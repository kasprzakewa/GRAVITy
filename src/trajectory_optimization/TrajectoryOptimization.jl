module TrajectoryOptimization

using CSV, DataFrames
using LinearAlgebra
using Printf
using DifferentialEquations: solve, ODEProblem, Vern9
using StaticArrays
using NearestNeighbors
using Plots

import ..CommonUtils: μ, r1_r2, u_eff, energy_integral_2d
import ..CommonUtils: newtonian_to_hamiltonian, hamiltonian_to_newtonian
import ..HamiltonianMethods: create_hamiltonian_problem, integrate_hamiltonian_method
import ..HamiltonianMethods: plot_hamiltonian_solution
import ..NewtonianMethods: create_newtonian_problem_with_function, solve_newtonian_problem

export load_orbit_data, compute_stm, extract_manifold_eigenvectors
export generate_manifold_initial_conditions, integrate_manifolds
export find_intersections, find_minimal_delta_v_earth_orbit
export plan_complete_trajectory, visualize_trajectory

function load_orbit_data(filepath::String)
    df = CSV.File(filepath) |> DataFrame
    x  = df[!, Symbol("X (LU)")]
    y  = df[!, Symbol("Y (LU)")]
    vx = df[!, Symbol("VX (LU/TU)")]
    vy = df[!, Symbol("VY (LU/TU)")]
    t  = df[!, Symbol("Time (TU)")]
    
    return x, y, vx, vy, t
end

function pcr3bp_with_stm!(du, u, μ, t)
    x, y, vx, vy = u[1:4]
    r1, r2 = r1_r2(x, y, μ)

    Ux = -(1 - μ)*(x + μ)/r1^3 - μ*(x - 1 + μ)/r2^3 + x
    Uy = -(1 - μ)*y/r1^3 - μ*y/r2^3 + y

    Uxx = 1 - (1 - μ)/r1^3 - μ/r2^3 +
          3*(1 - μ)*(x + μ)^2/r1^5 +
          3*μ*(x - 1 + μ)^2/r2^5

    Uyy = 1 - (1 - μ)/r1^3 - μ/r2^3 +
          3*(1 - μ)*y^2/r1^5 +
          3*μ*y^2/r2^5

    Uxy = 3*(1 - μ)*(x + μ)*y/r1^5 +
          3*μ*(x - 1 + μ)*y/r2^5

    A = [
        0   0    1     0
        0   0    0     1
        Uxx Uxy  0     2
        Uxy Uyy -2     0
    ]

    du[1] = vx
    du[2] = vy
    du[3] = 2*vy + Ux
    du[4] = -2*vx + Uy

    STM = reshape(u[5:end], 4, 4)
    dSTM = A * STM
    du[5:end] .= reshape(dSTM, 16)
end

function compute_stm(x0, y0, vx0, vy0, T; μ=μ)
    X0 = [x0, y0, vx0, vy0]
    STM0 = Matrix(I, 4, 4)
    u0 = vcat(X0, reshape(STM0, 16))
    
    println("Integrating variational equations to get STM over one period...")

    prob = create_newtonian_problem_with_function(u0, (0.0, T), pcr3bp_with_stm!)
    sol = solve_newtonian_problem(prob, Vern9(); dt=1e-3)
    
    println("Integration done.")
    
    uT = sol(T)
    STM_T = reshape(uT[5:end], 4, 4)
    eigen_vals, eigen_vecs = eigen(STM_T)
    
    println("\nEigenvalues of STM over one period:")
    println(eigen_vals)
    
    return STM_T, eigen_vals, eigen_vecs
end

function extract_manifold_eigenvectors(eigen_vals, eigen_vecs)
    idx_unstable = argmax(abs.(eigen_vals))
    v_unstable = eigen_vecs[:, idx_unstable]
    
    idx_stable = argmin(abs.(eigen_vals))
    v_stable = eigen_vecs[:, idx_stable]
    
    return v_unstable, v_stable
end

function generate_manifold_initial_conditions(orbit_points, v_unstable, v_stable, epsilon=1e-4)
    initial_unstable_pos = [point .+ epsilon * real(v_unstable[1:4]) for point in orbit_points]
    initial_unstable_neg = [point .- epsilon * real(v_unstable[1:4]) for point in orbit_points]
    
    initial_stable_pos = [point .+ epsilon * real(v_stable[1:4]) for point in orbit_points]
    initial_stable_neg = [point .- epsilon * real(v_stable[1:4]) for point in orbit_points]
    
    return initial_unstable_pos, initial_unstable_neg, initial_stable_pos, initial_stable_neg
end

function integrate_manifolds(initial_conditions, T, manifold_periods=5.0, forward=true)
    solutions = []
    
    dt_sign = forward ? 1e-3 : -1e-3
    tspan = forward ? (0.0, manifold_periods*T) : (0.0, -manifold_periods*T)
    
    for x0 in initial_conditions
        q_0, p_0 = newtonian_to_hamiltonian(x0)
        prob = create_hamiltonian_problem(q_0, p_0, tspan, dt_sign; μ=μ)
        sol = integrate_hamiltonian_method(prob, "LobattoIIIAIIIB_Order_6")
        push!(solutions, sol)
    end
    
    return solutions
end

function extract_xy(sol)
    xs = Tuple{Float64,Float64}[]
    ys = Tuple{Float64,Float64}[]
    for k in axes(sol.q, 1)
        x, y, vx, vy = hamiltonian_to_newtonian(sol.q[k], sol.p[k])
        push!(xs, (x, vx))
        push!(ys, (y, vy))
    end
    return xs, ys
end

function find_intersections(sol, R)
    points = SVector{2,Float64}[]
    meta = Tuple{Float64,Float64,Float64,Float64}[]
    
    for k in eachindex(sol.s.q.d)
        q = sol.s.q.d[k]
        p = sol.s.p.d[k]
        
        x, y = q[1], q[2]
        
        state = hamiltonian_to_newtonian(q, p)
        vx, vy = state[3], state[4]
        
        push!(points, SVector{2}(x + μ, y))
        push!(meta, (x, y, vx, vy))
    end
    
    tree = KDTree(points)
    idxs = inrange(tree, SVector{2}(0.0, 0.0), R)
    
    hits = [meta[i] for i in idxs]
    
    if !isempty(hits)
        minimal_distance = minimum([abs(R - sqrt((h[1] + μ)^2 + h[2]^2)) for h in hits])
        println("Found $(length(hits)) intersections with radius R = $R LU; minimal distance: $(round(minimal_distance, digits=8)) LU")
    else
        println("Found 0 intersections with radius R = $R LU")
    end
    
    return hits
end

function find_minimal_delta_v_earth_orbit(hits, R_LEO_LU)
    min_dv = Inf
    min_dv_position = nothing
    for (hx, hy, hvx, hvy) in hits
        vy = sqrt((1-μ)/R_LEO_LU)
        vx = sqrt(-vy^2 - 2*u_eff(hx, hy, μ) + 2*energy_integral_2d(hx, hy, hvx, hvy))
        v_orbit = [vx, vy]
        v_manifold = [hvx, hvy]
        dv = norm(v_orbit - v_manifold)
        if dv < min_dv
            min_dv = dv
            min_dv_position = (hx, hy, hvx, hvy)
        end
    end
    return min_dv, min_dv_position
end

function plan_complete_trajectory(stable_sol, unstable_sol, entry_position, exit_position, epsilon=1e-3)
    println("Finding intersections between stable and unstable manifolds near Lyapunov orbit...")
    
    candidates = []
    for i in eachindex(stable_sol.s.q.d)
        stable_state = hamiltonian_to_newtonian(stable_sol.s.q.d[i], stable_sol.s.p.d[i])
        unstable_state = hamiltonian_to_newtonian(unstable_sol.s.q.d[i], unstable_sol.s.p.d[i])
        
        dist = norm(stable_state[1:2] - unstable_state[1:2])
        if dist < epsilon
            dv = norm(stable_state[3:4] - unstable_state[3:4])
            push!(candidates, (i, dv, stable_state, unstable_state))
        end
    end
    
    if isempty(candidates)
        error("No intersections found between stable and unstable manifolds within epsilon = $epsilon")
    end
    
    min_idx, min_switch_dv, min_stable_state, min_unstable_state = 
        candidates[argmin([c[2] for c in candidates])]
    
    println("Found manifold intersection at index $min_idx with delta-v = $min_switch_dv LU/TU")
    
    return Dict(
        "switch_index" => min_idx,
        "switch_dv" => min_switch_dv,
        "stable_state" => min_stable_state,
        "unstable_state" => min_unstable_state,
        "entry_position" => entry_position,
        "exit_position" => exit_position
    )
end

function visualize_trajectory(stable_solutions, unstable_solutions, trajectory_plan, 
                             R_earth_LU, R_LEO_LU, orbit_data=nothing)
    p = plot(
        xlabel="x", ylabel="y",
        xlims=(-1, 1),
        ylims=(-1, 1),
        grid=true,
        aspect_ratio=:equal,
        legend=:topright,
        legendfontsize=7,
        legend_background_color=RGBA(1,1,1,0.8)
    )
    
    for sol in unstable_solutions
        plot_hamiltonian_solution(p, sol; color=:orangered, alpha=0.10, label=false)
    end
    for sol in stable_solutions
        plot_hamiltonian_solution(p, sol; color=:limegreen, alpha=0.10, label=false)
    end
    
    scatter!(p, [-μ], [0], color=:blue, markersize=6, label="Ziemia")
    
    θ = range(0, 2π, 400)
    leo_x = .-μ .+ R_LEO_LU .* cos.(θ)
    leo_y = R_LEO_LU .* sin.(θ)
    plot!(p, leo_x, leo_y, color=:black, linewidth=2, label="Orbita LEO")
    
    entry = trajectory_plan["entry_position"]
    exit = trajectory_plan["exit_position"]
    stable_state = trajectory_plan["stable_state"]
    switch_idx = trajectory_plan["switch_index"]
    
    stable_sol = trajectory_plan["stable_sol"]
    unstable_sol = trajectory_plan["unstable_sol"]
    
    entry_idx = 1
    min_dist_entry = Inf
    for i in eachindex(stable_sol.s.q.d)
        state = hamiltonian_to_newtonian(stable_sol.s.q.d[i], stable_sol.s.p.d[i])
        dist = sqrt((state[1] - entry[1])^2 + (state[2] - entry[2])^2)
        if dist < min_dist_entry
            min_dist_entry = dist
            entry_idx = i
        end
    end
    
    exit_idx = length(unstable_sol.s.q.d)
    min_dist_exit = Inf
    for i in eachindex(unstable_sol.s.q.d)
        state = hamiltonian_to_newtonian(unstable_sol.s.q.d[i], unstable_sol.s.p.d[i])
        dist = sqrt((state[1] - exit[1])^2 + (state[2] - exit[2])^2)
        if dist < min_dist_exit
            min_dist_exit = dist
            exit_idx = i
        end
    end
    
    stable_traj_x = Float64[]
    stable_traj_y = Float64[]
    for i in entry_idx:-1:switch_idx
        state = hamiltonian_to_newtonian(stable_sol.s.q.d[i], stable_sol.s.p.d[i])
        push!(stable_traj_x, state[1])
        push!(stable_traj_y, state[2])
    end
    if !isempty(stable_traj_x)
        plot!(p, stable_traj_x, stable_traj_y, color=:green, linewidth=2, 
              label="Trajektoria stabilna", linestyle=:solid)
    end
    
    unstable_traj_x = Float64[]
    unstable_traj_y = Float64[]
    for i in switch_idx:exit_idx
        state = hamiltonian_to_newtonian(unstable_sol.s.q.d[i], unstable_sol.s.p.d[i])
        push!(unstable_traj_x, state[1])
        push!(unstable_traj_y, state[2])
    end
    if !isempty(unstable_traj_x)
        plot!(p, unstable_traj_x, unstable_traj_y, color=:red, linewidth=2, 
              label="Trajektoria niestabilna", linestyle=:solid)
    end
    
    scatter!(p, [entry[1]], [entry[2]], 
        color=:green, markersize=8, markerstrokecolor=:black, markerstrokewidth=2, 
        label="Punkt wejścia")
    scatter!(p, [stable_state[1]], [stable_state[2]], 
        color=:orange, markersize=8, markerstrokecolor=:black, markerstrokewidth=2, 
        label="Punkt przejścia")
    scatter!(p, [exit[1]], [exit[2]], 
        color=:red, markersize=8, markerstrokecolor=:black, markerstrokewidth=2, 
        label="Punkt wyjścia")
    
    return p
end

end
