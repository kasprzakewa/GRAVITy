include("../src/CommonUtils.jl")
include("../src/NewtonianMethods.jl")
include("../src/HamiltonianMethods.jl")
include("../src/TrajectoryOptimization.jl")

using .TrajectoryOptimization: load_orbit_data, compute_stm, extract_manifold_eigenvectors,
                                generate_manifold_initial_conditions, integrate_manifolds,
                                find_optimal_heo_intersection, build_orbit_kdtree_weighted,
                                find_optimal_lyapunov_entry_weighted, find_optimal_lyapunov_exit_weighted,
                                visualize_trajectory
using Plots: savefig

function run_trajectory_optimization_example()
    println("="^80)
    println("TRAJECTORY OPTIMIZATION FOR EARTH-MOON SYSTEM")
    println("="^80)
    println()
    
    orbit_file = "example_data/lyapunov_orbit.csv"
    if !isfile(orbit_file)
        error("Plik z danymi orbity nie znaleziony: $orbit_file")
    end
    
    x, y, vx, vy, t = load_orbit_data(orbit_file)
    T = t[end]
    println("Loaded $(length(x)) orbit points, period T = $(round(T, digits=3)) TU\n")
    
    STM, eigenvalues, eigenvectors = compute_stm(x[1], y[1], vx[1], vy[1], T)
    v_unstable, v_stable = extract_manifold_eigenvectors(eigenvalues, eigenvectors)
    
    orbit_indices = 1:10:length(x)
    orbit_points = [[x[i], y[i], vx[i], vy[i]] for i in orbit_indices]
    
    epsilon = 1e-4
    iu_pos, iu_neg, is_pos, is_neg = generate_manifold_initial_conditions(
        orbit_points, v_unstable, v_stable, epsilon
    )
    
    manifold_periods = 5.0
    unstable_initial = vcat(iu_pos, iu_neg)
    unstable_solutions = integrate_manifolds(unstable_initial, T, manifold_periods, true)
    
    stable_initial = vcat(is_pos, is_neg)
    stable_solutions = integrate_manifolds(stable_initial, T, manifold_periods, false)
    
    println("Integrated $(length(stable_solutions)) stable manifolds and $(length(unstable_solutions)) unstable manifolds\n")
    
    R_earth_km = 6378.0
    HEO_alt_km = 106000.0
    LU_to_km = 384400.0
    TU_to_sec = 375200.0
    
    R_earth_LU = R_earth_km / LU_to_km
    R_HEO_LU = (R_earth_km + HEO_alt_km) / LU_to_km
    
    min_entry_dv = Inf
    min_entry_position = nothing
    min_entry_sol_idx = nothing
    
    for (idx, sol) in enumerate(stable_solutions)
        dv, position, count = find_optimal_heo_intersection(sol, R_HEO_LU)
        
        if position !== nothing && dv < min_entry_dv
            min_entry_dv = dv
            min_entry_position = position
            min_entry_sol_idx = idx
        end
    end
    
    if min_entry_position === nothing
        error("No intersections found on stable manifold with HEO!")
    end
    
    orbit_kdtree = build_orbit_kdtree_weighted(x, y, vx, vy; pos_weight=100.0)
    
    stable_sol = stable_solutions[min_entry_sol_idx]
    
    lyap_entry_dv, lyap_entry_manifold, lyap_entry_orbit, entry_checks = 
        find_optimal_lyapunov_entry_weighted(stable_sol, orbit_kdtree, x, y, vx, vy; 
                                             pos_weight=100.0, k_nearest=6, pos_tolerance=1e-3)
    
    lyap_exit_dv, best_unstable_idx, lyap_exit_manifold, lyap_exit_orbit, valid_count, exit_checks = 
        find_optimal_lyapunov_exit_weighted(unstable_solutions, orbit_kdtree, x, y, vx, vy, R_HEO_LU; 
                                            pos_weight=100.0, k_nearest=6, pos_tolerance=1e-3)
    
    if best_unstable_idx === nothing
        error("No intersections found on unstable manifold!")
    end
    
    unstable_sol = unstable_solutions[best_unstable_idx]
    min_exit_dv, min_exit_position, exit_count = find_optimal_heo_intersection(unstable_sol, R_HEO_LU)
    
    if min_exit_position === nothing
        error("No exit intersections found on unstable manifold!")
    end
    
    total_dv_m_s = (min_entry_dv + lyap_entry_dv + lyap_exit_dv + min_exit_dv) * LU_to_km / TU_to_sec * 1000
    
    println("="^80)
    println("RESULTS")
    println("="^80)
    println("Selected stable manifold: #$min_entry_sol_idx")
    println("Selected unstable manifold: #$best_unstable_idx (out of $(length(unstable_solutions)), $valid_count have HEO intersection)")
    println()
    println("Delta-v:")
    println("  HEO -> stable manifold:      $(round(min_entry_dv * LU_to_km / TU_to_sec * 1000, digits=2)) m/s")
    println("  stable manifold -> Lyapunov:          $(round(lyap_entry_dv * LU_to_km / TU_to_sec * 1000, digits=2)) m/s")
    println("  Lyapunov -> unstable manifold:        $(round(lyap_exit_dv * LU_to_km / TU_to_sec * 1000, digits=2)) m/s")
    println("  unstable manifold -> HEO:             $(round(min_exit_dv * LU_to_km / TU_to_sec * 1000, digits=2)) m/s")
    println("  ─────────────────────────────────────────")
    println("  SUMA:                      $(round(total_dv_m_s, digits=2)) m/s")
    println("="^80)
    println()
    
    try
        trajectory_plan = Dict(
            "entry_position" => min_entry_position,
            "lyap_entry_manifold" => lyap_entry_manifold,
            "lyap_entry_orbit" => lyap_entry_orbit,
            "lyap_exit_orbit" => lyap_exit_orbit,
            "lyap_exit_manifold" => lyap_exit_manifold,
            "exit_position" => min_exit_position,
            "stable_sol" => stable_sol,
            "unstable_sol" => unstable_sol,
            "sol_index" => min_entry_sol_idx
        )
        
        p = visualize_trajectory(
            stable_solutions, unstable_solutions, 
            trajectory_plan, R_earth_LU, R_HEO_LU,
            (x, y)
        )
        
        mkpath("results/trajectory_optimization")
        savefig(p, "results/trajectory_optimization/optimal_trajectory.png")
        println("Visualization saved to: results/trajectory_optimization/optimal_trajectory.png\n")
    catch e
        println("Failed to generate visualization: $e\n")
    end
end

function main_trajectory()
    try
        run_trajectory_optimization_example()
    catch e
        println("\nERROR: $e")
        showerror(stdout, e, catch_backtrace())
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_trajectory()
end
