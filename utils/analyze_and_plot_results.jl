using CSV
using DataFrames
using Plots
using Statistics
using Printf

gr()
default(fontfamily="Computer Modern", framestyle=:box)

function load_results(filepath::String)
    if !isfile(filepath)
        @warn "File does not exist: $filepath"
        return nothing
    end
    return CSV.read(filepath, DataFrame)
end

function parse_test_case(test_case)
    test_case_str = String(test_case)
    
    if occursin("E < E_L1", test_case_str)
        return "below_L1"
    elseif occursin("E_L1 < E < E_L2", test_case_str)
        return "between_L1_L2"
    elseif occursin("E_L4 < E", test_case_str)
        return "above_L4"
    else
        return "unknown"
    end
end

function get_base_method_name(method_name)
    return replace(String(method_name), "_" => " ")
end

function plot_fixed_T(df::DataFrame, energy_level::String, T::Float64, output_dir::String, 
                      metric::Symbol, metric_label::String;
                      exclude_methods::Vector{String}=String[])
    
    subset_df = filter(row -> begin
        el = parse_test_case(row.test_case)
        t = Float64(row.T)
        el == energy_level && t == T
    end, df)
    
    if nrow(subset_df) == 0
        @warn "No data for $energy_level, T=$T"
        return
    end
    
    subset_df.base_method = [get_base_method_name(m) for m in subset_df.method]
    
    if !isempty(exclude_methods)
        subset_df = filter(row -> !(row.base_method in exclude_methods), subset_df)
    end
    
    if nrow(subset_df) == 0
        return
    end
    
    sort!(subset_df, :dt)
    methods = unique(subset_df.base_method)
    
    energy_level_display = replace(energy_level, "_" => " ")
    p = plot(xlabel="dt", ylabel=metric_label, 
             title="$energy_level_display, T=$T",
             xscale=:log10, yscale=:log10,
             legend=:outertopright,
             size=(800, 600))
    
    for method in methods
        method_data = filter(row -> row.base_method == method, subset_df)
        if nrow(method_data) > 0
            plot!(p, method_data.dt, method_data[!, metric], 
                  label=method, marker=:circle, linewidth=2)
        end
    end
    
    mkpath(output_dir)
    filename = joinpath(output_dir, "$(energy_level)_T$(Int(T))_$(metric).png")
    savefig(p, filename)
    println("Saved: $filename")
end

function plot_fixed_dt(df::DataFrame, energy_level::String, dt::Float64, output_dir::String, 
                       metric::Symbol, metric_label::String;
                       exclude_methods::Vector{String}=String[])
    
    subset_df = filter(row -> begin
        el = parse_test_case(row.test_case)
        method_dt = Float64(row.dt)
        el == energy_level && abs(method_dt - dt) < 1e-10
    end, df)
    
    if nrow(subset_df) == 0
        @warn "No data for $energy_level, dt=$dt"
        return
    end
    
    subset_df.base_method = [get_base_method_name(m) for m in subset_df.method]
    
    if !isempty(exclude_methods)
        subset_df = filter(row -> !(row.base_method in exclude_methods), subset_df)
    end
    
    if nrow(subset_df) == 0
        return
    end
    
    sort!(subset_df, :T)
    methods = unique(subset_df.base_method)
    
    energy_level_display = replace(energy_level, "_" => " ")
    p = plot(xlabel="Final time T", ylabel=metric_label, 
             title="$energy_level_display, dt=$dt",
             legend=:outertopright,
             size=(800, 600))
    
    for method in methods
        method_data = filter(row -> row.base_method == method, subset_df)
        if nrow(method_data) > 0
            plot!(p, method_data.T, method_data[!, metric], 
                  label=method, marker=:circle, linewidth=2)
        end
    end
    
    mkpath(output_dir)
    filename = joinpath(output_dir, "$(energy_level)_dt$(dt)_$(metric).png")
    savefig(p, filename)
    println("Saved: $filename")
end

function create_heatmap(df::DataFrame, T::Float64, output_dir::String, 
                        metric::Symbol, metric_label::String;
                        exclude_methods::Vector{String}=String[],
                        exclude_variants::Vector{String}=String[])

    subset_df = filter(row -> Float64(row.T) == T, df)

    if nrow(subset_df) == 0
        @warn "No data for T=$T"
        return
    end

    subset_df.energy_level = [parse_test_case(row.test_case) for row in eachrow(subset_df)]
    subset_df.base_method = [get_base_method_name(m) for m in subset_df.method]
    
    if !isempty(exclude_methods)
        subset_df = filter(row -> !(row.base_method in exclude_methods), subset_df)
    end
    
    if nrow(subset_df) == 0
        return
    end

    subset_df.variant = ["$(row.energy_level)_T$(Int(T))_dt$(row.dt)" for row in eachrow(subset_df)]
    
    if !isempty(exclude_variants)
        subset_df = filter(row -> !(row.variant in exclude_variants), subset_df)
    end
    
    if nrow(subset_df) == 0
        return
    end

    variants = sort(unique(subset_df.variant))
    methods = sort(unique(subset_df.base_method))

    heatmap_data = zeros(length(methods), length(variants))

    for (i, method) in enumerate(methods)
        for (j, variant) in enumerate(variants)
            matching_rows = filter(r -> r.base_method == method && r.variant == variant, subset_df)
            heatmap_data[i, j] = nrow(matching_rows) > 0 ? matching_rows[1, metric] : NaN
        end
    end

    variants_display = [replace(v, "_" => " ") for v in variants]
    methods_display = methods

    vmin, vmax = extrema(skipmissing(vec(heatmap_data)))
    threshold = (vmin + vmax) / 2

    p = heatmap(variants_display, methods_display, heatmap_data,
        xlabel="Variant (Energy level + T + dt)",
        ylabel="Method",
        title="$metric_label dla T=$T",
        color=cgrad(:matter),
        size=(1400, 800),
        xrotation=45,
        bottom_margin=15Plots.mm,
        left_margin=10Plots.mm,
        colorbar=false)

    for i in 1:length(methods_display)
        for j in 1:length(variants_display)
            value = heatmap_data[i, j]
            if !isnan(value)
                value_str = if abs(value) < 0.001
                    @sprintf("%.2e", value)
                elseif abs(value) < 1.0
                    @sprintf("%.4f", value)
                else
                    @sprintf("%.3f", value)
                end
                txt_color = value > threshold ? :white : :black
                annotate!(p, j - 0.5, i - 0.5,
                    text(value_str, 7, txt_color, :center))
            end
        end
    end

    mkpath(output_dir)
    filename = joinpath(output_dir, "heatmap_T$(Int(T))_$(metric).png")
    savefig(p, filename)
    println("Saved: $filename")
end

function process_results(csv_file::String, output_base_dir::String, method_type::String;
                        exclude_methods::Vector{String}=String[], 
                        exclude_variants::Vector{String}=String[],
                        subfolder::String="")
    println("\n=== Processing results: $method_type $(isempty(subfolder) ? "" : "($subfolder)") ===")
    
    df = load_results(csv_file)
    if df === nothing
        return
    end
    
    println("Loading $(nrow(df)) rows of data")
    
    required_cols = ["method", "test_case", "dt", "T"]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        @error "Missing columns in CSV: $missing_cols"
        return
    end
    
    energy_levels = sort(unique([parse_test_case(tc) for tc in df.test_case]))
    T_values = sort(unique(df.T))
    dt_values = sort(unique(df.dt))
    
    println("Energy levels: $energy_levels")
    println("T values: $T_values")
    println("dt values: $dt_values")
    if !isempty(exclude_methods)
        println("Excluded methods: $exclude_methods")
    end
    if !isempty(exclude_variants)
        println("Excluded variants: $exclude_variants")
    end
    
    metrics = [
        (:max_energy_drift, "Maximum energy drift"),
        (:mean_energy_drift, "Mean energy drift"),
        (:std_energy_drift, "Energy drift std dev"),
        (:execution_time_s, "Execution time (s)"),
        (:memory_mb, "Memory (MB)")
    ]
    
    base_dir = isempty(subfolder) ? output_base_dir : joinpath(output_base_dir, subfolder)
    
    println("\nCreating plots for fixed T...")
    for energy_level in energy_levels
        for T in T_values
            output_dir = joinpath(base_dir, energy_level, "fixed_T")
            for (metric, label) in metrics
                plot_fixed_T(df, energy_level, T, output_dir, metric, label,
                            exclude_methods=exclude_methods)
            end
        end
    end
    
    println("\nCreating plots for fixed dt...")
    for energy_level in energy_levels
        for dt in dt_values
            output_dir = joinpath(base_dir, energy_level, "fixed_dt")
            for (metric, label) in metrics
                plot_fixed_dt(df, energy_level, dt, output_dir, metric, label,
                             exclude_methods=exclude_methods)
            end
        end
    end
    
    println("\nCreating heatmaps...")
    for T in T_values
        output_dir = joinpath(base_dir, "heatmaps")
        for (metric, label) in metrics
            create_heatmap(df, T, output_dir, metric, label,
                          exclude_methods=exclude_methods,
                          exclude_variants=exclude_variants)
        end
    end
    
    println("\n=== Completed processing: $method_type $(isempty(subfolder) ? "" : "($subfolder)") ===")
end

function analyze_and_plot()
    println("=== Numerical methods results analysis ===\n")
    
    hamiltonian_csv = "results/hamiltonian_methods/hamiltonian_methods_results.csv"
    newtonian_csv = "results/newtonian_methods/newtonian_methods_results.csv"
    
    hamiltonian_output = "results/plots/hamiltonian_methods"
    newtonian_output = "results/plots/newtonian_methods"
    
    if isfile(hamiltonian_csv)
        process_results(hamiltonian_csv, hamiltonian_output, "Hamiltonian methods", 
                       subfolder="all_methods")
        
        process_results(hamiltonian_csv, hamiltonian_output, "Hamiltonian methods",
                       exclude_methods=["Explicit Runge-Kutta method with Heun2 tableau", "LobattoIIIC 4th order method", "Implicit midpoint"],
                       exclude_variants=["between_L1_L2_T100_dt0.01"],
                       subfolder="selected_methods")
    else
        @warn "File not found: $hamiltonian_csv"
    end
    
    if isfile(newtonian_csv)
        process_results(newtonian_csv, newtonian_output, "Newtonian methods",
                       subfolder="all_methods")
        
        process_results(newtonian_csv, newtonian_output, "Newtonian methods",
                       exclude_methods=["Euler method", "Adams-Bashforth 5th order method"],
                       exclude_variants=["between_L1_L2_T100_dt0.01", "above_L4_T100_dt0.01"],
                       subfolder="selected_methods")
    else
        @warn "File not found: $newtonian_csv"
    end
    
    println("\n=== Analysis complete ===")
    println("Results saved in results/plots/ folder")
end

if abspath(PROGRAM_FILE) == @__FILE__
    analyze_and_plot()
end
