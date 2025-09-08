using Distributed
addprocs(2)

# import Pkg; Pkg.add("Parameters")
# import Pkg; Pkg.add("Plots")

### add @everywhere in front of this
@everywhere using Parameters, Plots, SharedArrays #import the libraries we want
include("4 parallel_stochastic_growth_functions.jl") #import the functions that solve our growth model
@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!

### Unpack results and make plots
@unpack val_func, pol_func = res
@unpack k_grid, Z_grid, nZ = prim

############## Make plots
#### value function over K for each Z

p_S_V = plot(
    title = "Value Function V(K,Z)",
    xlabel = "capital K",
    ylabel = "value V(K,Z)",
    legend = :bottomright,
    dpi = 300
)
for z_index in 1:nZ
    plot!(p_S_V, k_grid,
        val_func[:, z_index], 
        label=z_index == 1 ? "Z_g: Good productivity" : "Z_b: Bad productivity")
end

p_S_V

# savefig only takes (figure, filename); dpi is set above in the plot attributes
savefig(p_S_V, "4 Parallel_Stochastic_Value_Functions.png")

#########################################
#### policy functions

p_S_P = plot(
    title = "Policy Function K'(K,Z)",
    ylabel = "policy K'(K,Z)",
    label = "policy K'(K,Z)",
    xlabel = "capital K",
    legend = :bottomright,
    linestyle = :solid,
    dpi = 300
)

for z_index in 1:nZ
    plot!(p_S_P, k_grid,
        pol_func[:, z_index], 
        label = z_index == 1 ? "Z_g: Good productivity" : "Z_b: Bad productivity"
        )
end

plot!(p_S_P,
    k_grid, k_grid, label = "45 degree", color = "red", linestyle = :dash)

p_S_P

savefig(p_S_P, "4 Parallel_Stochastic_Policy_Functions.png")

#########################################
#### changes in policy function
pol_func_δ = pol_func .- k_grid

p_S_PFC = plot(
    title = "Saving Policy Function K'(K,Z) - K",
    ylabel = "saving policy K'(K,Z) - K",
    label = "",
    xlabel = "capital K",
    legend = :bottomleft,
    dpi = 300
)

for z_index in 1:nZ
    plot!(p_S_PFC, k_grid, 
        pol_func_δ[:, z_index], 
        label = z_index == 1 ? "Z_g: Good productivity" : "Z_b: Bad productivity"
        )
end

hline!(p_S_PFC, 
    [0], linestyle = :dash, color = :black, label = "")

p_S_PFC

savefig(p_S_PFC, "4 Parallel_Stochastic_Policy_Functions_Changes.png")

println("All done!")
#########################################################################
