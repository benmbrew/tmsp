#import base_model_sparse
#import policy_iteration_sparse
import numpy as np

# Given a problem instance I for our MDP, determines the minimum number of
#   ALS ambulances required for the average utilization of an ambulance to be
#   less than u. We consider only the policy that admits every call. We can
#   leverage the MDP code that we've already written, by deleting Action 0
#   unavailable whenever Action 1 is available.

def find_budget(I, u):
    fleet_size=0
    util=1

    while util > u:
        I.NA = fleet_size
        I.NB = 0
        I.L = I.lamH + I.lamL + (I.NA + I.NB)*I.mu

        # Construct MDP instance
        S, R, A, P =  base_model_sparse.build_mdp(I, False)

        # Modified action space: accepting (when possible) is the only policy
        for s in S:
            if A[s] == [0, 1]:
                A[s] = [1]

        # Obtain stationary distribution induced by this policy
        _, _, _, pi, _ = policy_iteration_sparse.solve_mdp(S, R, A, P)
        util, _, _     = getParams(I, A, u, pi)
        
        
    return fleet_size*I.CA

# An alternate version of the above function, where instead the termination
# criterion is not utilization-based, but instead finds the minimum number of
# ambulances needed so that calls can be serviced immediately with prob. p.

def find_budget_alternate(I, p):
    fleet_size = 0
    prob       = 0

    while prob < p:
        fleet_size += 1
        
        I.NA = fleet_size
        I.NB = 0
        I.L  = I.lamH + I.lamL + (I.NA + I.NB)*I.mu

        # Construct "modified" MDP instance
        S, R, A, P = base_model_sparse.build_mdp(I, False)
        for s in S:
            if A[s] == [0, 1]:
                A[s] = [1]

        # Obtain stationary distribution induced by this policy
        _, _, _, pi, _ = policy_iteration_sparse.solve_mdp(S, R, A, P)
        prob = float(1 - pi[-1])

        print(fleet_size, prob)
        
    return fleet_size*I.CA

# Given a problem instance I, as well as the stationary distribution of a
#   Markov chain (induced by a given policy) outputs:
#
# Hsvc : High-priority service level, fraction of hi-priority calls served
#           by an ALS unit in the long run
# Lsvc : High-priority service level, fraction of hi-priority calls served
#           (and not redirected, for whatever reason) in the long run
#
# We make graphs with these!

def getMDPparams(I, A, u, pi):
    # If no ALS or BLS units, utilization set to 1. These values won't be
    #   updated in the code below.
    Hsvc = 0
    Lsvc = 0

    # Hsvc = Fraction of time at least one ALS unit free
    # Lsvc = Fraction of time at least one BLS unit free, or ALS unit free
    #           and policy does not redirect low-priority calls
    for s in range(len(pi)):
        i, j = I.lookup[s]
        if i < I.NA:
            Hsvc += pi[s]
            if j == I.NB and u[s]:
                Lsvc += pi[s]

        if j < I.NB:
            Lsvc += pi[s]
    
    return Hsvc, Lsvc


# Given a problem instance I, as well as the stationary distribution of a
#   Markov chain (induced by a given policy) outputs:
#
# pA : Long-run utilization of ALS ambulances
# pB : Long-run utilization of BLS ambulances
#  f : Fraction of low-priority calls receiving a response from ALS units
#       when all BLS units are busy
# Thease are used as inputs for the integer program.

def getLPparams(I, A, u, pi):
    # If no ALS or BLS units, utilization set to 1. These values won't be
    #   updated in the code below.
    pA = (I.NA == 0)
    pB = (I.NB == 0)
    
    # Fraction of time ALS unit responds to low-priority call, as well as
    #   the fraction of time where such a decision needs to be made
    num   = 0.0
    denom = 0.0

    for s in range(len(pi)):
        i, j  = I.lookup[s]

        if I.NA > 0:
            pA += i*pi[s]/I.NA
            
        if I.NB > 0:
            pB += j*pi[s]/I.NB

        if 1 in A[s]:
            denom += pi[s]
            if u[s] == 1:
                num += pi[s]

    try:
        f = num/denom
    except:
        f = 0

    return [pA, pB, f]


