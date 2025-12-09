"""
Comprehensive test runner for newtonian and hamiltonian methods.

Executes all numerical integration methods on various test cases and generates
comparative analysis of energy conservation and performance.
"""

using Dates
using ..CommonUtils: get_test_cases, TestResult, save_results_to_csv

"""Run all newtonian and hamiltonian method tests with analysis."""
function run_comprehensive_tests()
    println("="^100)
    println("PCR3BP NUMERICAL METHODS TESTING")
    println("="^100)
    println("Start time: $(get_timestamp())")
    println("Test cases: $(length(get_test_cases())) different energy scenarios")
    println("Testing both Newtonian and Hamiltonian interpretations")
    println()
    
    println("PHASE 1: NEWTONIAN METHODS")
    println("="^50)
    newtonian_results = run_newtonian_tests()
    
    println("\n" * "="^50)
    println("PHASE 2: HAMILTONIAN METHODS") 
    println("="^50)
    hamiltonian_results = run_hamiltonian_tests()
    
    all_results = vcat(newtonian_results, hamiltonian_results)
    
    println("\n" * "="^100)
    println("COMPARISON AND ANALYSIS")
    println("="^100)
    
    generate_comprehensive_analysis(newtonian_results, hamiltonian_results)
    
    mkpath("results")
    save_results_to_csv(all_results, "results/pcr3bp_results.csv")
    
    println("\nTest completion time: $(get_timestamp())")
    println("="^100)
    
    return all_results
end

"""Get current timestamp as string."""
function get_timestamp()
    return string(Dates.now())
end

"""Generate comparison between newtonian and hamiltonian methods."""
function generate_comprehensive_analysis(newtonian_results, hamiltonian_results)
    println("\nPERFORMANCE COMPARISON:")
    println("-"^60)
    
    valid_newtonian = filter(r -> isfinite(r.max_energy_drift), newtonian_results)
    valid_hamiltonian = filter(r -> isfinite(r.max_energy_drift), hamiltonian_results)
    all_valid = vcat(valid_newtonian, valid_hamiltonian)
    
    if !isempty(valid_newtonian)
        best_newtonian = valid_newtonian[argmin([r.max_energy_drift for r in valid_newtonian])]
        println("Best Newtonian method:")
        println("  $(best_newtonian.method_name) on $(best_newtonian.case_name)")
        println("  Energy drift: |ΔE| = $(best_newtonian.max_energy_drift)")
        println("  Execution time: $(best_newtonian.execution_time)s")
    end
    
    if !isempty(valid_hamiltonian)
        best_hamiltonian = valid_hamiltonian[argmin([r.max_energy_drift for r in valid_hamiltonian])]
        println("Best Hamiltonian method:")
        println("  $(best_hamiltonian.method_name) on $(best_hamiltonian.case_name)")
        println("  Energy drift: |ΔE| = $(best_hamiltonian.max_energy_drift)")
        println("  Execution time: $(best_hamiltonian.execution_time)s")
    end

    println("\nPERFORMANCE BY TEST CASE:")
    println("-"^60)
    
    test_cases = get_test_cases()
    for test_case in test_cases
        case_name = test_case["name"]
        println("\nCase: $case_name ($(test_case["description"]))")
        
        case_results = filter(r -> r.case_name == case_name && isfinite(r.max_energy_drift), all_valid)
        if !isempty(case_results)
            sorted_case = sort(case_results, by=r -> r.max_energy_drift)
            
            println("  Best 3 methods:")
            for (i, result) in enumerate(sorted_case[1:min(3, length(sorted_case))])
                method_clean = result.method_name
                println("    $i. $method_clean: |ΔE| = $(result.max_energy_drift)")
            end
        else
            println("  No successful methods for this case")
        end
    end 
    
    if !isempty(all_valid)
        println("\nTOP 10 METHODS BY ENERGY CONSERVATION:")
        println("-"^60)
        
        method_case_best = Dict{Tuple{String,String}, TestResult}()
        
        for result in all_valid
            base_method = result.method_name
            key = (base_method, result.case_name)
            
            if !haskey(method_case_best, key) || 
               result.max_energy_drift < method_case_best[key].max_energy_drift
                method_case_best[key] = result
            end
        end
        
        best_results = collect(values(method_case_best))
        sorted_results = sort(best_results, by=r -> r.max_energy_drift)
        
        for (i, result) in enumerate(sorted_results[1:min(10, length(sorted_results))])
            method_clean = result.method_name
            method_type = contains(method_clean, "Gauss") || contains(method_clean, "Lobatto") || 
                         contains(method_clean, "Symplectic") || contains(method_clean, "Implicit") ? 
                         "Hamiltonian" : "Newtonian"
            
            println("$i. $method_clean ($method_type, $(result.case_name)): |ΔE| = $(result.max_energy_drift), Time = $(result.execution_time)s")
        end
    end
    

end

"""Main entry point with command-line argument handling."""
function main_tests()
    args = ARGS
    
    if "--newtonian" in args
        println("Running newtonian methods only...")
        run_newtonian_tests()
    elseif "--hamiltonian" in args
        println("Running hamiltonian methods only...")
        run_hamiltonian_tests()
    else
        println("Running comprehensive tests...")
        run_comprehensive_tests()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_tests()
end