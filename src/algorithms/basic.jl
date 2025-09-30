module BasicAlgorithms

function greet(name::String)
    return "Hello, " * name * "!"
end

function add(a::Number, b::Number)
    return a + b
end

function fibonacci(n::Int)
    if n <= 1
        return n
    else
        return fibonacci(n-1) + fibonacci(n-2)
    end
end

export greet, fibonacci, add

end