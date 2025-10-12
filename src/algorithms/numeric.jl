module CR3BPNumeric

using DifferentialEquations
using Plots

export cr3bp!, plot_cr3bp

default(
    background_color = RGB(0.05, 0.05, 0.07),
    foreground_color = :white,
    guidefont = font("Helvetica", 12, "bold"),
    tickfont = font("Helvetica", 10),
    legendfont = font("Helvetica", 10),
    titlefont = font("Helvetica", 14, "bold"),
    grid = false,
    size = (800, 600),
    dpi = 300
)

function cr3bp!(du, u, p, t)
    μ = p
    x, y, z, x1, y1, z1 = u
    
    r1² = (x + μ)^2 + y^2 + z^2
    r2² = (x - (1 - μ))^2 + y^2 + z^2
    r1³ = r1²^(3/2)
    r2³ = r2²^(3/2)
    
    grav_x = -(1 - μ) * (x + μ) / r1³ - μ * (x - (1 - μ)) / r2³
    grav_y = -(1 - μ) * y / r1³ - μ * y / r2³
    grav_z = -(1 - μ) * z / r1³ - μ * z / r2³
    
    du[1] = x1
    du[2] = y1
    du[3] = z1
    du[4] = 2 * y1 + x + grav_x
    du[5] = -2 * x1 + y + grav_y
    du[6] = grav_z
end

u0 = [0.8, 0.1, 0.0, 0.0, 0.1, 0.0]
tspan = (0.0, 100.0)
μ = 0.01215
prob = ODEProblem(cr3bp!, u0, tspan, μ)

sol_euler = solve(prob, Euler(), dt=0.001)
sol_rk4 = solve(prob, RK4(), reltol=1e-8, abstol=1e-8)
sol_vernet = solve(prob, Vern9(), reltol=1e-10, abstol=1e-10)

function plot_cr3bp(sol, method_name, μ; color=:cyan)
    plt = plot(
        sol, vars=(1, 2),
        title = "Trajektoria w CR3BP - $(method_name)",
        xlabel = "x",
        ylabel = "y",
        legend = :bottomright,
        linecolor = color,
        linewidth = 2.0,
        aspect_ratio = 1
    )

    scatter!([-μ], [0], color=:blue, label="m1 (Ziemia)", markersize=6)
    scatter!([1 - μ], [0], color=:orange, label="m2 (Księżyc)", markersize=5)
    
    plot!(plt, framestyle=:box)
    return plt
end

plt1 = plot_cr3bp(sol_euler, "Euler", μ, color=:magenta)
savefig(plt1, "trajectory_euler_dark.png")

plt2 = plot_cr3bp(sol_rk4, "RK4", μ, color=:lime)
savefig(plt2, "trajectory_rk4_dark.png")

plt3 = plot_cr3bp(sol_vernet, "Vern9 (Adaptive RK)", μ, color=:cyan)
savefig(plt3, "trajectory_vernet_dark.png")

end
