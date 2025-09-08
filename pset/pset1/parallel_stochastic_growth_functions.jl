#keyword-enabled structure to hold model primitives

# @everywhere using Parameters, SharedArrays

### Our goal: parallelize computing

@everywhere @with_kw struct Primitives
    ####
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 90.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::SharedVector{Float64} = collect(range(start=k_min, stop=k_max, length=nk))

    #### Add stochastic technology shocks
    nZ::Int64 = 2 #number of technology states {Z_g, Z_b}
    Z_grid::SharedVector{Float64} = [1.25, 0.2] #supp of technology states
    Π::SharedMatrix{Float64} = [0.977 0.023; 
                                0.074 0.926] #transition matrix of state Z_g and Z_b
end

#structure that holds model results for k and Z
@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64, 2} #value function
    pol_func::SharedArray{Float64, 2} #policy function
end

#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives() #initialize primtives 
    val_func = SharedArray{Float64,2}(zeros(prim.nk, prim.nZ))
    pol_func = SharedArray{Float64,2}(zeros(prim.nk, prim.nZ))
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

####################################################

# Goal: we want to parallelize capital levels

#Bellman Operator

function Bellman(prim::Primitives,res::Results)
    # @unpack val_func = res #unpack value function
    @unpack_Results res #unpack results struct
    @unpack_Primitives prim #unpack model primitives
    # v_next = zeros(prim.nk, prim.nZ)
    v_next = SharedArray{Float64}((prim.nk, prim.nZ))

    # We need to comment out this line of setting choice_lower    
    # choice_lower = 1 #for exploiting monotonicity of policy function


    ########### Here is where we should parallelize the computing
    #### How? Use @distributed

    ########### Task: add stochastic technology shocks
    
    @sync @distributed for z_index in 1:nZ
        Z = Z_grid[z_index] #loop over technology states
        @sync @distributed for k_index in 1:nk
            k = k_grid[k_index] #loop over capital grid
            candidate_max = -Inf #bad candidate max
            budget = Z * k^α + (1-δ)*k #budget

            for kp_index in 1:nk #loop over possible selections of k'   
                c = budget - k_grid[kp_index] #consumption given k' selection
                if c > 0 #check for positivity
                    EV = 0.0 #initialize expected value
                    for zp_index in 1:nZ
                        prob = Π[z_index, zp_index] #transition prob
                        value = val_func[kp_index, zp_index] #value function
                        EV += prob * value #expected value of next period's value function
                    end
                    val = log(c) + β * EV #compute value
                    if val > candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[k_index, z_index] = k_grid[kp_index] #update policy function
                        # choice_lower = kp_index #update lowest possible choice
                    end
                end
            end
            v_next[k_index, z_index] = candidate_max #update value function
        end
    end
    return v_next #return next guess of value function
end

#Value function iteration (VFI)
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0 #counter
    while err > tol #begin iteration
        ## The iteration
        v_next = Bellman(prim, res) #spit out new vectors 
        err = maximum(abs.(v_next .- res.val_func)) #/abs(v_next[prim.nk, 1]) #reset error level
        res.val_func .= v_next #update value function
        n += 1
    end
    println("With parallel, Stochastic Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
