module TrajectoryOptimization

using CSV, DataFrames
using LinearAlgebra
using DifferentialEquations: solve, ODEProblem, Vern9
using NearestNeighbors
using Plots

import ..CommonUtils: μ, r1_r2, u_eff, energy_integral_2d
import ..CommonUtils: find_lagrange_points
import ..NewtonianMethods: create_newtonian_problem_with_function, create_newtonian_problem, solve_newtonian_problem

export load_orbit_data, compute_stm, extract_manifold_eigenvectors
export generate_manifold_initial_conditions, integrate_manifolds
export find_optimal_heo_intersection
export build_orbit_kdtree_weighted, find_optimal_lyapunov_entry_weighted, find_optimal_lyapunov_exit_weighted
export visualize_trajectory

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
    
    prob = create_newtonian_problem_with_function(u0, (0.0, T), pcr3bp_with_stm!)
    sol = solve_newtonian_problem(prob, Vern9(); dt=1e-3)
    
    uT = sol(T)
    STM_T = reshape(uT[5:end], 4, 4)
    eigen_vals, eigen_vecs = eigen(STM_T)
    
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
        prob = create_newtonian_problem(x0, tspan)
        sol = solve_newtonian_problem(prob, Vern9(); dt=abs(dt_sign))
        push!(solutions, sol)
    end
    
    return solutions
end

function build_orbit_kdtree_weighted(orbit_x, orbit_y, orbit_vx, orbit_vy; pos_weight=100.0)
    M = length(orbit_x)
    
    data = zeros(4, M)
    for i in 1:M
        data[1, i] = orbit_x[i] * pos_weight
        data[2, i] = orbit_y[i] * pos_weight
        data[3, i] = orbit_vx[i]
        data[4, i] = orbit_vy[i]
    end
    
    return KDTree(data)
end

function find_optimal_lyapunov_entry_weighted(stable_sol, orbit_kdtree, 
                                              orbit_x, orbit_y, orbit_vx, orbit_vy;
                                              pos_weight=100.0, k_nearest=6, pos_tolerance=1e-3)
    M = length(orbit_x)
    min_dv = Inf
    min_manifold_position = nothing
    min_orbit_position = nothing
    total_checks = 0
    
    for state in stable_sol.u
        mx, my, mvx, mvy = state[1], state[2], state[3], state[4]
        
        query = [mx * pos_weight, my * pos_weight, mvx, mvy]
        idxs, dists = knn(orbit_kdtree, query, min(k_nearest, M))
        
        for orbit_idx in idxs
            total_checks += 1
            
            lx = orbit_x[orbit_idx]
            ly = orbit_y[orbit_idx]
            lvx = orbit_vx[orbit_idx]
            lvy = orbit_vy[orbit_idx]
            
            pos_error = sqrt((mx - lx)^2 + (my - ly)^2)
            
            if pos_error < pos_tolerance
                dv = norm([mvx - lvx, mvy - lvy])
                
                if dv < min_dv
                    min_dv = dv
                    min_manifold_position = (mx, my, mvx, mvy)
                    min_orbit_position = (lx, ly, lvx, lvy)
                end
            end
        end
    end
    
    return min_dv, min_manifold_position, min_orbit_position, total_checks
end

function find_optimal_lyapunov_exit_weighted(unstable_solutions, orbit_kdtree,
                                             orbit_x, orbit_y, orbit_vx, orbit_vy, R_HEO_LU;
                                             pos_weight=100.0, k_nearest=6, pos_tolerance=1e-3)
    
    M = length(orbit_x)
    min_dv_global = Inf
    best_manifold_idx = nothing
    best_manifold_position = nothing
    best_orbit_position = nothing
    
    valid_manifolds = 0
    total_checks = 0
    
    for (traj_idx, sol) in enumerate(unstable_solutions)
        has_heo_intersection = false
        for state in sol.u
            x, y = state[1], state[2]
            dist_to_earth = sqrt((x + μ)^2 + y^2)
            if abs(dist_to_earth - R_HEO_LU) < 0.01
                has_heo_intersection = true
                break
            end
        end
        
        if !has_heo_intersection
            continue
        end
        valid_manifolds += 1
        
        for state in sol.u
            mx, my, mvx, mvy = state[1], state[2], state[3], state[4]
            
            query = [mx * pos_weight, my * pos_weight, mvx, mvy]
            idxs, dists = knn(orbit_kdtree, query, min(k_nearest, M))
            
            for orbit_idx in idxs
                total_checks += 1
                
                lx = orbit_x[orbit_idx]
                ly = orbit_y[orbit_idx]
                lvx = orbit_vx[orbit_idx]
                lvy = orbit_vy[orbit_idx]
                
                pos_dist = sqrt((mx - lx)^2 + (my - ly)^2)
                
                if pos_dist < pos_tolerance
                    dv = norm([mvx - lvx, mvy - lvy])
                    
                    if dv < min_dv_global
                        min_dv_global = dv
                        best_manifold_idx = traj_idx
                        best_manifold_position = (mx, my, mvx, mvy)
                        best_orbit_position = (lx, ly, lvx, lvy)
                    end
                end
            end
        end
    end
    
    return min_dv_global, best_manifold_idx, best_manifold_position, best_orbit_position, valid_manifolds, total_checks
end

function find_optimal_heo_intersection(sol, R_HEO_LU; tolerance=0.01)
    min_dv = Inf
    min_dv_position = nothing
    intersection_count = 0
    
    for state in sol.u
        x, y, vx, vy = state[1], state[2], state[3], state[4]
        distance = sqrt((x + μ)^2 + y^2)
        
        if abs(distance - R_HEO_LU) < tolerance
            intersection_count += 1
            
            vy_orbit = sqrt((1-μ)/R_HEO_LU)
            vx_orbit = sqrt(-vy_orbit^2 - 2*u_eff(x, y, μ) + 2*energy_integral_2d(x, y, vx, vy))
            v_orbit = [vx_orbit, vy_orbit]
            v_manifold = [vx, vy]
            dv = norm(v_orbit - v_manifold)
            
            if dv < min_dv
                min_dv = dv
                min_dv_position = (x, y, vx, vy)
            end
        end
    end
    
    return min_dv, min_dv_position, intersection_count
end

function visualize_trajectory(stable_solutions, unstable_solutions, trajectory_plan, 
                             R_earth_LU, R_HEO_LU, orbit_data=nothing)
    p = plot(
        xlims=(-1, 2),
        ylims=(-1, 1.5),
        grid=true,
        aspect_ratio=:equal,
        legend=:topright,
    )

    for sol in unstable_solutions
        xs = [state[1] for state in sol.u]
        ys = [state[2] for state in sol.u]
        plot!(p, xs, ys, color=:red, alpha=0.10, label=false)
    end
    
    for sol in stable_solutions
        xs = [state[1] for state in sol.u]
        ys = [state[2] for state in sol.u]
        plot!(p, xs, ys, color=:limegreen, alpha=0.10, label=false)
    end
    
    scatter!(p, [-μ], [0], color=:dodgerblue, markersize=8, label="Earth", markerstrokecolor=:black, markerstrokewidth=2)
    scatter!(p, [1-μ], [0], color=:lightgray, markersize=6, label="Moon", markerstrokecolor=:black, markerstrokewidth=2)
    
    L_points = find_lagrange_points()
    L1_x = L_points["L1"][1]
    
    θ = range(0, 2π, 400)
    heo_x = .-μ .+ R_HEO_LU .* cos.(θ)
    heo_y = R_HEO_LU .* sin.(θ)
    plot!(p, heo_x, heo_y, color="#221732", linewidth=2, label="HEO")
    
    entry = trajectory_plan["entry_position"]
    exit = trajectory_plan["exit_position"]
    
    lyap_entry_manifold = get(trajectory_plan, "lyap_entry_manifold", nothing)
    lyap_entry_orbit = get(trajectory_plan, "lyap_entry_orbit", nothing)
    lyap_exit_orbit = get(trajectory_plan, "lyap_exit_orbit", nothing)
    lyap_exit_manifold = get(trajectory_plan, "lyap_exit_manifold", nothing)
    
    stable_sol = trajectory_plan["stable_sol"]
    unstable_sol = trajectory_plan["unstable_sol"]
    
    entry_idx = argmin([sqrt((state[1] - entry[1])^2 + (state[2] - entry[2])^2) for state in stable_sol.u])
    
    lyap_entry_idx = entry_idx
    if lyap_entry_manifold !== nothing
        lyap_entry_idx = argmin([sqrt((state[1] - lyap_entry_manifold[1])^2 + (state[2] - lyap_entry_manifold[2])^2) for state in stable_sol.u])
    end
    
    lyap_exit_idx = 1
    if lyap_exit_manifold !== nothing
        lyap_exit_idx = argmin([sqrt((state[1] - lyap_exit_manifold[1])^2 + (state[2] - lyap_exit_manifold[2])^2) for state in unstable_sol.u])
    end
    
    exit_idx = argmin([sqrt((state[1] - exit[1])^2 + (state[2] - exit[2])^2) for state in unstable_sol.u])
    
    stable_traj_x = Float64[]
    stable_traj_y = Float64[]
    for i in entry_idx:-1:lyap_entry_idx
        state = stable_sol.u[i]
        push!(stable_traj_x, state[1])
        push!(stable_traj_y, state[2])
    end
    if !isempty(stable_traj_x)
        plot!(p, stable_traj_x, stable_traj_y, color=:limegreen, linewidth=3, 
              label="Stable trajectory", linestyle=:solid)
    end
    
    unstable_traj_x = Float64[]
    unstable_traj_y = Float64[]
    for i in lyap_exit_idx:exit_idx
        state = unstable_sol.u[i]
        push!(unstable_traj_x, state[1])
        push!(unstable_traj_y, state[2])
    end
    if !isempty(unstable_traj_x)
        plot!(p, unstable_traj_x, unstable_traj_y, color=:red, linewidth=3, 
              label="Unstable trajectory", linestyle=:solid)
    end
    
    scatter!(p, [entry[1]], [entry[2]], 
        color=:limegreen, markersize=6, markerstrokecolor=:black, markerstrokewidth=2, 
        label="Stable manifold entry point")
    
    if lyap_entry_manifold !== nothing
        scatter!(p, [lyap_entry_manifold[1]], [lyap_entry_manifold[2]], 
            color=:orange, markersize=6, markerstrokecolor=:black, markerstrokewidth=2, 
            label="Lyapunov orbit entry point")
    end
    
    if lyap_exit_manifold !== nothing
        scatter!(p, [lyap_exit_manifold[1]], [lyap_exit_manifold[2]], 
            color=:yellow, markersize=6, markerstrokecolor=:black, markerstrokewidth=2, 
            label="Lyapunov orbit exit point")
    end
    
    scatter!(p, [exit[1]], [exit[2]], 
        color=:red, markersize=6, markerstrokecolor=:black, markerstrokewidth=2, 
        label="Unstable manifold exit point")
    
    return p
end

end
