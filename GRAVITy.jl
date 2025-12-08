include("src/CommonUtils.jl")
include("src/NewtonianMethods.jl")
include("src/HamiltonianMethods.jl")
include("src/TrajectoryOptimization.jl")

module GRAVITyMain

using ..NewtonianMethods
using ..HamiltonianMethods

include("utils/run_tests.jl")
include("utils/run_trajectory_optimization.jl")
include("utils/analyze_and_plot_results.jl")

function display_usage()
    println("""
Usage:
  julia GRAVITy.jl [command] [options]

Commands:
  test                Run numerical methods tests
  trajectory          Run trajectory optimization
  analyze             Analyze and plot test results
  help                Show this help message

Test Options:
  --newtonian         Run only Newtonian methods tests
  --hamiltonian       Run only Hamiltonian methods tests
  (default: run both Newtonian and Hamiltonian)

Examples:
  julia GRAVITy.jl test                    # Run all numerical tests
  julia GRAVITy.jl test --newtonian        # Run only Newtonian tests  
  julia GRAVITy.jl test --hamiltonian      # Run only Hamiltonian tests
  julia GRAVITy.jl trajectory              # Run trajectory optimization
  julia GRAVITy.jl analyze                 # Analyze and plot results
  julia GRAVITy.jl help                    # Show this help
""")
end

function main_gravity(args=ARGS)
    if isempty(args)
        display_usage()
        return
    end
    
    command = args[1]
    
    if command == "test"
        if "--newtonian" in args
            println("Running Newtonian methods only...")
            NewtonianMethods.run_newtonian_tests()
        elseif "--hamiltonian" in args
            println("Running Hamiltonian methods only...")
            HamiltonianMethods.run_hamiltonian_tests()
        else
            println("Running tests...")
            run_comprehensive_tests()
        end
    elseif command == "trajectory"
        run_trajectory_optimization_example()
    elseif command == "analyze"
        analyze_and_plot()
    elseif command == "help"
        display_usage()
    else
        println("Error: Unknown command '$command'")
        println("Run 'julia GRAVITy.jl help' for usage information")
    end
end

end 

if abspath(PROGRAM_FILE) == @__FILE__
    GRAVITyMain.main_gravity(ARGS)
end
