using Parameters, Plots, Printf #import the libraries
default_blue = theme_palette(:default)[1]
default_orange = theme_palette(:default)[2]
include("pset2_functions.jl") #import the functions that solve our Huggett model
prim, res = Initialize() #initialize primitive and results structs

# Solve the model!
# elapse = @elapsed Solve_model(prim, res) 
@printf("Time: %0.3f seconds to solve this Huggett model (tol level = 10^-4).\n", float(@elapsed Solve_model(prim, res)))

### Unpack results and make plots
@unpack val_func, pol_func = res
@unpack β, α, S, nS, Π, a_min, a_max, na, a_grid = prim #unpack model primitives

###########################################################
####     Part I: Computing Complete market EQM         ####
###########################################################

# Stationary transition matrix

Π_stationary = Calculate_Π_stationary(prim, res)


###########################################################
####     Part II: Computing Incomplete market EQM      ####
###########################################################

############## Make plots

employment_labels = ["employed", "unemployed"]


#### 1 Value function 

V = plot(
    title = "Value Function V(a,S)",
    xlabel = "asset position a",
    ylabel = "value V(a,S)",
    legend = :bottomright,
    dpi=300
)

for s_index in 1:nS
    plot!(V, a_grid,
        val_func[:, s_index],
        label="S = $(S[s_index]) ($(employment_labels[s_index]))"
    )
end

V

# savefig(V, "output/1 Value_Functions.png")


#########################################

#### 2 policy functions (over a for each S)

#### 2.1 Find threshold asset levels where g(a,s) = a

thresholds = Find_thresholds(prim, res)
println("Threshold asset levels where g(a,s) = a:")
for s_index in 1:nS
    println("a*_$(employment_labels[s_index]) = $(round(thresholds[s_index], digits=4))\n")
end
#########################################

P = plot(
        title = "Policy Function g(a,S)",
        ylabel = "policy a' = g(a,S)",
        label = "policy g(a,S)",
        xlabel = "asset position a",
        legend = :best,
        linestyle = :solid,
        dpi = 300
)

for s_index in 1:nS
    plot!(P, a_grid,
        pol_func[:, s_index],
        label="S = $(S[s_index]) ($(employment_labels[s_index]))"
    )
end

plot!(P, 
a_grid, a_grid, label = "45 degree (g(a,S)=a)", color = "gray", linestyle = :dash)

# Add vertical lines at threshold points
vline!(P, [thresholds[1]], label="\$\\hat{a}_{$(employment_labels[1])} = $(round(thresholds[1], digits=4))\$", color=default_blue, linestyle=:dash)

vline!(P, [thresholds[2]], label = "\$\\hat{a}_{$(employment_labels[2])} = $(round(thresholds[2], digits=4))\$", color=default_orange, linestyle=:dash)


P
# savefig(P, "output/2 Policy_Functions.png")


#########################################

### 3 Wealth Distribution

println("=== WEALTH DISTRIBUTION ===")

wealth = Calculate_wealth(prim, res)

# check
println("Total mass in wealth distribution: ", sum(wealth))
println("Max density - Employed: ", maximum(wealth[:, 1]))
println("Max density - Unemployed: ", maximum(wealth[:, 2]))
println("Employed peak at wealth level: ", a_grid[argmax(wealth[:, 1])])
println("Unemployed peak at wealth level: ", a_grid[argmax(wealth[:, 2])])

# Plot Wealth distribution

W = plot(
    title="Wealth Distribution",
    ylabel="density",
    label="Wealth",
    xlabel="wealth = y(s) + a",
    legend=:best,
    linestyle=:solid,
    dpi=300
)

for s_index in 1:nS
    plot!(W, a_grid, wealth[:, s_index],
        label="S = $(S[s_index]) ($(employment_labels[s_index]))"
    )
end

W
# savefig(W, "output/3 Wealth_Distributions.png")


#########################################
### 4 Lorenz Curve and Gini Coefficient
println(" ")
println("=== LORENZ CURVE AND GINI COEFFICIENT ===")

cdf_w, cum_w = Calculate_lorenz(wealth)


# Lorenz Curve

L = plot(
    title="Lorenz curve",
    ylabel="Cumulative wealth",
    xlabel="Fraction of agents",
    legend=:best,
    linestyle=:solid,
    dpi=300
)

plot!(L, 
    cdf_w, [cum_w cdf_w],
    labels=["Lorenz Curve" "Perfect Equality"],
    color = [default_blue "gray"],
)

# savefig(L, "output/4 Lorenz_Curve.png")

# Gini Coefficient
gini_coefficient = Calculate_gini(prim, cdf_w, cum_w)


#########################################
####   Part III: Welfare Analysis    ####
#########################################

###########
### Benchmark: complete markets
# Recall: We already calculated stationary transition matrix in Part I 

Π_e = Π_stationary[1] # stationary distribution of employed agents
Π_u = Π_stationary[2] # stationary distribution of unemployed agents
Π_stationary
println(" ")
println("=== Welfare in complete markets (First Best) ===")
w_FB = Calculate_W_FB(prim, Π_stationary)

@printf("w_FB (rounded to 5 digits): %0.5f.", float(w_FB))
println(" ")

###########
### Counterfactual: incomplete markets

# We calculate "Consumption Equivalent λ"
# We have a closed form expression for λ in slides!

λ_inc = Calculate_consumption_equivalence(prim, res, w_FB)

Consumption_equivalence = plot(
    title="Consumption equivalence λ(a,S)",
    xlabel="asset position a",
    ylabel="CE: λ(a,S)",
    legend=:best,
    dpi=300
)

for s_index in 1:nS
    plot!(Consumption_equivalence, a_grid,
        λ_inc[:, s_index],
        label="S = $(S[s_index]) ($(employment_labels[s_index]))"
    )
end

hline!(Consumption_equivalence, 
    [0], linestyle = :dash, color = :red, label = "")

Consumption_equivalence
# savefig(Consumption_equivalence, "output/5 Consumption_Equivalence.png")

## REMARK: if λ(a,s) > 0, agent is better off in complete markets economy 
##              (i.e., need insurance, or favor change to complete markets)
##         if λ(a,s) < 0, agent is worse off in complete markets economy


##################################################
## What is the welfare in the incomplete market counterfactual?
println(" ")
println("=== Welfare in incomplete markets (Counterfactual) ===")
w_inc = Calculate_W_inc(prim, res)
@printf("w_inc (rounded to 5 digits): %0.5f.", float(w_inc))
println(" ")
println("REMARK: This alignes to class slides, 'Aggregate welfare is higher in the complete markets economy than the incomplete markets economy (no surprise)'.")
println(" ")
## REMARK: aggregate welfare is higher in the complete markets economy than the incomplete markets economy (no surprise)

## “what fraction of consumption would a person in a steady state of the incomplete markets environment be willing to pay (if positive) or have to be paid (if negative) in all future periods to achieve the allocation of the complete markets environment?”
## Welfare gain/loss

WG = Calculate_Welfare_gain(prim, res, λ_inc)
println(" ")
@printf("Welfare gain of switching to complete markets (rounded to 5 digits): %0.5f.", float(WG))

fraction_of_favoring_complete_markets = Calculate_favor_Complete(prim, res, λ_inc)
println(" ")
@printf("Fraction of agents that favor switching to complete markets (rounded to 5 digits): %0.5f.", float(fraction_of_favoring_complete_markets))

############################################
