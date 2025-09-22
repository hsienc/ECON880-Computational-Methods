#keyword-enabled structure to hold model primitives

## ECON 880 pset2

### Our goal: Solve a GE model that generates an 
### endogenous steady state wealth distribution (Huggett, 1993)

@with_kw struct Primitives
    #### Parameters choices
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #RRA
    S::Vector{Float64} = [1, 0.5] #earning states (employed/unemployed)
    nS::Int64 = length(S)

    #### Decision rule for asset position
    a_min::Float64 = -2 #asset position lower bound (borrowing constraint)    
    a_max::Float64 = 2.5 #asset position upper bound (*I follow this choice in slide instead of "5" in pset description)
    na::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64, 1} = collect(range(start=a_min, stop=a_max, length=na))

    #### Markov transition matrix
    Π::Matrix{Float64} = [0.97 0.03; 
                          0.5 0.5] 
end



#structure that holds model results for EQM computation
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
    μ::Array{Float64, 2} #distribution over (a,s)
    q::Float64 #price of non-contingent bond; assume q \in [0,1] and β < q -> β < q <= 1 
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() 
    val_func = [zeros(prim.na) zeros(prim.na)]
    pol_func = [zeros(prim.na) zeros(prim.na)]

    ### Competitive EQM benchmark: q = β = 0.9932
    ### My guess: q should slightly higher than β due to incomplete markets
    q = prim.β + (1-prim.β)/8  # To make it slightly higher than β
    μ = [ones(prim.na) ones(prim.na)]./(prim.na*2) #initial guess of distribution μ

    ###
    res = Results(val_func, pol_func, μ, q) #initialize results struct
    prim, res #return deliverables
end

####################################################

# Find stationary eqm (& distribution)

function Calculate_Π_stationary(prim::Primitives, res::Results)
    @unpack Π, S, nS = prim

    # Simple algebraic solution for 2x2 transition matrix

    if nS == 2
        # Direct formula for 2×2 case
        π₁₂ = Π[1, 2]  # transition from employed to unemployed
        π₂₁ = Π[2, 1]  # transition from unemployed to employed

        π_employed = π₂₁ / (π₁₂ + π₂₁)
        π_unemployed = π₁₂ / (π₁₂ + π₂₁)

        π_stationary = [π_employed, π_unemployed]

        println("=== STATIONARY DISTRIBUTION (2×2, by Detailed Balance Equations) ===")
        println("Π₁₂ (employed → unemployed): ", π₁₂)
        println("Π₂₁ (unemployed → employed): ", π₂₁)
        println("π[employed] = π₂₁/(π₁₂+π₂₁) = $π₂₁/$(π₁₂+π₂₁) = $(round(π_employed, digits=5))")
        println("π[unemployed] = π₁₂/(π₁₂+π₂₁) = $π₁₂/$(π₁₂+π₂₁) = $(round(π_unemployed, digits=5))")
        println("Sum (should be 1.0): ", sum(π_stationary))
        println("Verification π*Π ≈ π: ", isapprox(π_stationary' * Π, π_stationary', atol=1e-10))
        println("==============================================")

    else
        # General solution for n×n case using linear system
        # Solve (Π' - I)π = 0 with constraint that sum(π) = 1
        M = Π' - I
        M[end, :] .= 1.0  # Replace last row with normalization constraint
        b = zeros(nS)
        b[end] = 1.0
        π_stationary = M \ b

        println("=== STATIONARY DISTRIBUTION (General) ===")
        for s_index = 1:nS
            println("π[$s_index] = $(round(π_stationary[s_index], digits=5))")
        end
        println("Sum (should be 1.0): ", sum(π_stationary))
        println("========================================")
    end

    return π_stationary
end


############### ################# ############### 
############### Part II - Step a1  ############### 
############### ################# ############### 

#### Inner loop ####

# T_Bellman Operator
# We solve for the value and policy functions given q

function Bellman(prim::Primitives,res::Results)
    @unpack val_func, q = res #unpack results struct
    @unpack β, α, S, nS, Π, a_min, a_max, na, a_grid = prim #unpack model primitives

    val_func_next = [zeros(na) zeros(na)]
    pol_func_next = [zeros(na) zeros(na)]

    ########### Task: Solve Household problem

    #### Loop over employment states
    for s_index in 1:nS
        s = S[s_index] #current employment state
        #### Given states, loop over asset grid
        choice_lower = 1 #for exploiting monotonicity of policy function
        for a_index in 1:na
            a = a_grid[a_index] #current asset position
            val_func_next[a_index, s_index] = -Inf 
            for ap_index in choice_lower:na #loop over possible selections of a'
                budget = s + a 
                c = budget - q*a_grid[ap_index] #consumption given a' selection
                if c > 0 #check for positivity consumption
                    # Current period utility
                    u = (c^(1 - α) - 1) / (1 - α) #CRRA utility function
                    
                    # EV of tomorrow's value function
                    EV = u
                    for sp_index in 1:nS
                        prob = Π[s_index, sp_index] #transition probability
                        value = val_func[ap_index, sp_index] #value function at (a', s')
                        EV += β * prob * value #expected value of next period value function
                    end

                    if EV > val_func_next[a_index, s_index] #check for new max value
                        val_func_next[a_index, s_index] = EV #update value function
                        pol_func_next[a_index, s_index] = a_grid[ap_index] #update policy function
                        choice_lower = ap_index #update lowest possible choice
                    end
                end
            end
        end
    end
    val_func_next, pol_func_next #return next guess of value and policy functions
end

### Solve Household Problem
function Solve_HH(prim::Primitives, res::Results; tol=1e-4)
    err = 1
    n = 0 # counter

    while true
        val_func_next, pol_func_next = Bellman(prim, res) #spit out new vectors 
        err = maximum(abs.(val_func_next .- res.val_func)) #reset error level
        res.val_func = val_func_next #update value function
        res.pol_func = pol_func_next #update policy function
        n += 1
        if err < tol #check convergence
            break
        end
    end
    println("[HH Problem] Value function converged in ", n, " iterations.")
    println(" ")
end

############### ################# ############### 
############### Part II - Step a2  ############### 
############### ################# ############### 

###### T_star_Bellman Operator
# 
# Goal: Updates the invariant structure


function T_star_Bellman(prim::Primitives, res::Results; tol::Float64=1e-4)
    @unpack pol_func, μ = res
    @unpack na, nS, Π, a_grid = prim  

    μ_next = zeros(na, nS)  # A × S  

    ### Loop over all states (a,s) ∈ Ã × S
    for s_index = 1:nS          
        for a_index = 1:na      
            
            ### Get optimal asset choice from policy function
            ap = pol_func[a_index, s_index]
            
            ### Search the grids; χ
            ap_index = argmin(abs.(ap .- a_grid))
            
            ### Update distribution of future states
            for sp_index = 1:nS
                μ_next[ap_index, sp_index] += Π[s_index, sp_index] * μ[a_index, s_index]
            end
        end
    end
    return μ_next
end

### Solve invariant structure
function Solve_Invariant(prim::Primitives, res::Results; tol::Float64=1e-4)
    err = 1.0
    n = 0 # counter

    while true
        μ_next = T_star_Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(μ_next .- res.μ)) #reset error level
        res.μ = μ_next #update μ distribution
        n += 1
        if err < tol #check convergence
            break
        end
    end
    println("[Invariant] Invariant structure converged in ", n, " iterations.")
    println(" ")
end

############### ################# ############### 
############### Part II - Step b  ############### 
############### ################# ###############   

### Market Clearing: calculate excess demand for bonds and update price q

## Recall: β < q <= 1 

function Update_q(results::Results; tol::Float64=1e-4)
    @unpack a_grid, β = Primitives()
    @unpack μ, q = results

    # Calculate net asset holdings (should be zero in equilibrium)
    excess_demand = sum(μ[:, 1] .* a_grid) + sum(μ[:, 2] .* a_grid)

    if abs(excess_demand) > tol
        if excess_demand < 0
            # (-) ED: too much borrowing; need to lower bond price and move towards "β"
            q_update = q + excess_demand * (q - β)/2
            println("ED is negative: ", excess_demand)
            println("Excess demand for borrowing: ", float(-excess_demand))
            println("-> we need to lower price from ", q, " down to ", q_update)
            println(" ")
        elseif excess_demand > 0
            # (+) ED: too much saving; need to raise bond price and move towards "1"
            q_update = q + excess_demand * (1 - q)/2
            println("ED is positive: ", excess_demand)
            println("Excess demand for bond purchase: ", float(excess_demand))
            println("-> we need to raise price from ", q, " up to ", q_update)
            println(" ")
        end
        results.q = q_update
        return(false)
    else
        println("ED is within tolerance, with: ", excess_demand)
        println("-> Bond price converges to: ", q)
        println("-> Asset/bond Market clears!")
        println(" ")
        return(true)
    end
end

### Solve the entire model
# Market clearing iteration

function Solve_model(prim::Primitives, res::Results)
    n = 1 #counter
    while true
        println("== Market Clearing Iteration ", n, " ==")

        ### Given prices, use Bellman operator to solve HH problem 
        Solve_HH(prim, res)

        ### Given policy functions, find invariant
        Solve_Invariant(prim, res)
        
        ### Update price and check market clearing conditions
        mkt_clear = Update_q(res)

        if mkt_clear
            println("Market clears in ", n, " iterations")
            println(" ")
            println("=============================================")
            println(" ")
            return
        end
        
        n += 1
    end
end

### Find threshold asset levels where g(a,s) = a
function Find_thresholds(prim::Primitives, res::Results)
    @unpack pol_func = res
    @unpack a_grid, nS = prim
    
    thresholds = zeros(nS)
    for s_index in 1:nS
        policy_diff = pol_func[:, s_index] .- a_grid

        ### Search the grid; minimum difference in policy functions
        min_idx = argmin(abs.(policy_diff))
        thresholds[s_index] = a_grid[min_idx]
    end
    return thresholds
end


############### ################# ############### 
############### Part II - Step c  ############### 
############### ################# ###############  

## Wealth distribution, Gini Coefficient, and Lorenz curve

## From class slides:
# The the steady state wealth distribution in Figure 5 uses wealth
# defined as cash from direct deposit of y (s) plus net bond holdings
# a (spike at y (e) + a). Spikes arise because some fraction of
# households stuck on the borrowing constraint receive an
# employment opportunity and save.

function Calculate_wealth(prim::Primitives, res::Results)
    @unpack μ = res
    @unpack a_grid, na, S, nS = prim

    # Initialize wealth distribution matrix
    w = [zeros(na) zeros(na)]

    # Calculate wealth distribution: wealth = income y(s) + assets (a)
    for s_index = 1:nS
        for a_index = 1:na
            total_wealth = S[s_index] + a_grid[a_index]
            w_index = argmin(abs.(total_wealth .- a_grid))
            w[w_index, s_index] = μ[a_index, s_index]
        end
    end
    w
end

function Calculate_lorenz(wealth::Array{Float64,2}) 
    @unpack a_grid, na, nS = Primitives()
    
    cdf_w = zeros(na)
    cum_w = zeros(na)
    
    # Calculate cumulative distribution function (CDF) and cumulative wealth
    # Loop over asset grid to get population and wealth shares
    for a_index = 1:na-1
        for s_index = 1:nS
            # Population mass
            cdf_w[a_index] += wealth[a_index, s_index] 
            # Wealth mass   
            cum_w[a_index] += wealth[a_index, s_index] * a_grid[a_index] 
        end
        cdf_w[a_index+1] = cdf_w[a_index]
        cum_w[a_index+1] = cum_w[a_index]
    end
    
    # Handle the last grid point
    for s_index = 1:nS
        cdf_w[na] += wealth[na, s_index]
        cum_w[na] += wealth[na, s_index] * a_grid[na]
    end
    
    # Normalize cumulative wealth for Lorenz curve
    # Calculate fraction of total wealth owned by the fraction of population
    cum_w = cum_w ./ maximum(cum_w)

    return cdf_w, cum_w
end

############    Part II - Step c ############



# The Gini coefficient with respect to total wealth

function Calculate_gini(prim::Primitives, cdf_w::Vector{Float64}, cum_w::Vector{Float64})
    @unpack na, a_grid = prim
    
    # Calculate Gini coefficient using trapezoidal rule
    # Gini = 1 - 2 * (area under Lorenz curve)
    area_under = 0.0
    
    for i = 1:na-1
        # Trapezoidal rule: (1/2) * (y1 + y2) * (x2 - x1)
        area_under += 0.5 * (cum_w[i] + cum_w[i+1]) * (cdf_w[i+1] - cdf_w[i])
    end

    gini_coefficient = 1.0 - 2.0 * area_under

    println("=== GINI CALCULATION ===")
    println("Area under Lorenz curve: ", area_under)
    println("Gini coefficient: ", gini_coefficient)
    @printf("Rounded to 5 digits: %0.5f.", float(gini_coefficient))
    println(" ")
    println("Expected range: [0, 1] where 0=perfect equality, 1=perfect inequality")
    println("===================================")

    return gini_coefficient
end


############### ################### ############### 
############### Part III - Welfare  ############### 
############### ################### ############### 

# Need: stationary distribution, value function, primitives

function Calculate_W_FB(prim::Primitives, Π_stationary::Vector{Float64})
    @unpack α, β, Π, S = prim
    ## Construct consumption (stationary eqm)
    # Π_stationary = Calculate_Π_stationary(prim, res)
    Π_e = Π_stationary[1] # stationary distribution of employed agents
    Π_u = Π_stationary[2] # stationary distribution of unemployed agents
    ##
    c_FB = Π_e * S[1] + Π_u * S[2] # full insurance consumption
    u_FB = (c_FB^(1 - α) - 1) / (1 - α) # full insurance utility
    w_FB = u_FB/(1 - β) # full insurance lifetime utility
    return w_FB
end


function Calculate_consumption_equivalence(prim::Primitives, res::Results, w_FB::Float64)
    @unpack β, α, na, nS = prim
    @unpack val_func, μ = res
    
    ## Find λ from the closed form expression in slides
    numerator = w_FB + 1 / ((1 - α) * (1 - β))
    denominator = res.val_func .+ 1 / ((1 .- α) .* (1 .- β))
    λ_incomplete = (numerator ./ denominator).^(1 / (1 .- α)) .- 1

    return λ_incomplete
end


function Calculate_W_inc(prim::Primitives, res::Results)
    @unpack β, α, Π, S, nS, na = prim
    @unpack val_func, μ = res

    W_inc = 0.0

    for a_index = 1:na
        for s_index = 1:nS
            W_inc += μ[a_index, s_index] * val_func[a_index, s_index]
        end
    end
    W_inc
end

function Calculate_Welfare_gain(prim::Primitives, res::Results, λ_incomplete::Array{Float64,2})
    @unpack β, α, Π, S, nS, na = prim
    @unpack val_func, μ = res
    WG = 0.0
    for a_index = 1:na
        for s_index = 1:nS
            WG += μ[a_index, s_index] * λ_incomplete[a_index, s_index]
        end
    end
    WG
end

function Calculate_favor_Complete(prim::Primitives, res::Results, λ_incomplete::Array{Float64,2})

    ## REMARK: if λ(a,s) >= 0, agent is better off in complete markets economy 
    ##              (i.e., need insurance, or favor change to complete markets)
    ##         if λ(a,s) < 0, agent is worse off in complete markets economy

    @unpack β, α, Π, S, nS, na = prim
    @unpack val_func, μ = res
    fraction = 0.0
    for a_index = 1:na
        for s_index = 1:nS
            favor = (λ_incomplete[a_index, s_index] >= 0) ? 1 : 0
            fraction += μ[a_index, s_index] * favor
        end
    end
    fraction
end








##############################################################################
# Pset 1 codes
# #Value function iteration (VFI)
# function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
#     n = 0 #counter
#     while err > tol #begin iteration
#         ## The iteration
#         v_next = Bellman(prim, res) #spit out new vectors 
#         err = maximum(abs.(v_next .- res.val_func)) #/abs(v_next[prim.nk, 1]) #reset error level
#         res.val_func .= v_next #update value function
#         n += 1
#     end
#     println("Without parallel, Stochastic Value function converged in ", n, " iterations.")
# end

# #solve the model
# function Solve_model(prim::Primitives, res::Results)
#     V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
# end
##############################################################################
