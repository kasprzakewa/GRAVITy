using HTTP
using JSON3
using Sockets
using LinearAlgebra

DEV_MODE = get(ENV, "DEV", "0") == "1"

if DEV_MODE
    try
        using Revise
        println("Development mode: reloading enabled")
    catch e
        println("Could not load Revise (continuing without reload): $e")
    end
else
    println("Production mode")
end

const PORT = 8001

const GLOBAL_CONTEXT = Module()

try
    include("../algorithms/basic.jl")
    using .BasicAlgorithms
    
    exported_names = names(BasicAlgorithms)
    loaded_algorithms = []
    
    for name in exported_names
        try
            func = getfield(BasicAlgorithms, name)
            
            if isa(func, Function)
                Core.eval(GLOBAL_CONTEXT, :($(Symbol(name)) = $func))
                push!(loaded_algorithms, string(name))
            end
        catch e
            println("Warning: Could not load algorithm $name: $e")
        end
    end
    
    println("Algorithms module loaded successfully")
    println("Available algorithms: $(join(loaded_algorithms, ", "))")
    
catch e
    println("Error loading algorithms module: $e")
end

function execute_julia_code(code::String)
    result = nothing
    output = ""
    error_msg = nothing
    
    try
        result = Core.eval(GLOBAL_CONTEXT, Meta.parse(code))
        output = ""
        
    catch exec_error
        error_msg = string(exec_error)
    end
    
    return Dict(
        "success" => error_msg === nothing,
        "error" => error_msg,
        "output" => output,
        "result" => result
    )
end

function handle_execute(req::HTTP.Request)
    try
        body = String(req.body)
        data = JSON3.read(body)
        
        if !haskey(data, "code")
            response_body = JSON3.write(Dict("error" => "Missing 'code' field in request"))
            return HTTP.Response(400, ["Content-Type" => "application/json"], response_body)
        end
        
        code = data.code
        result = execute_julia_code(code)
        
        response_body = JSON3.write(result)
        return HTTP.Response(200, ["Content-Type" => "application/json"], response_body)
        
    catch e
        println("Error in handle_execute: $e")
        error_response = JSON3.write(Dict("error" => "Server error: $(string(e))"))
        return HTTP.Response(500, ["Content-Type" => "application/json"], error_response)
    end
end

function handle_health(req::HTTP.Request)
    return HTTP.Response(200, 
        ["Content-Type" => "application/json"], 
        JSON3.write(Dict(
            "status" => "healthy",
            "service" => "julia-repl",
            "version" => string(VERSION)
        ))
    )
end

function handle_reset(req::HTTP.Request)
    try
        for name in names(GLOBAL_CONTEXT, all=true)
            if name != :eval && name != :include
                try
                    Core.eval(GLOBAL_CONTEXT, :($(Symbol(name)) = nothing))
                catch
                    # ignore errors when trying to reset built-in names
                end
            end
        end
        
        return HTTP.Response(200, 
            ["Content-Type" => "application/json"], 
            JSON3.write(Dict("status" => "context reset"))
        )
    catch e
        return HTTP.Response(500, JSON3.write(Dict("error" => "Reset error: $(string(e))")))
    end
end

function router(req::HTTP.Request)
    if req.method == "POST" && req.target == "/execute"
        return handle_execute(req)
    elseif req.method == "GET" && req.target == "/health"
        return handle_health(req)
    elseif req.method == "POST" && req.target == "/reset"
        return handle_reset(req)
    elseif req.method == "GET" && req.target == "/"
        return HTTP.Response(200, 
            ["Content-Type" => "application/json"], 
            JSON3.write(Dict(
                "service" => "Julia REPL Microservice",
                "endpoints" => [
                    "GET /health - Health check",
                    "POST /execute - Execute Julia code",
                    "POST /reset - Reset REPL context"
                ]
            ))
        )
    else
        return HTTP.Response(404, JSON3.write(Dict("error" => "Not found")))
    end
end

function start_server()
    println("Starting Julia REPL microservice on port $PORT...")
    println("Available endpoints:")
    println("  GET  /health     - Health check")
    println("  POST /execute    - Execute Julia code")
    println("  POST /reset      - Reset REPL context")
    println("  GET  /           - Service info")
    
    HTTP.serve(router, "0.0.0.0", PORT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_server()
end