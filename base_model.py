import numpy as np
# No buffer or queue

# --------------------------------
### This script has the code for the building the mdp (matrices necessary for policy iteration)
# --------------------------------

class instance:
    pass

def build_mdp(I, debug):
    # Create dicionaries to lookup / reverse lookup elements in state space
    I.lookup, I.revlookup = build_dicts(I)
    N = len(I.lookup) # total number of statess
    S = range(N) # range of states (object that you iterate through).

    # Uniformization constant, uniformized parameters
    # arrival rates and service completion divided by the maximal rate
    lh = float(I.lamH)/I.L
    ll = float(I.lamL)/I.L
    mu = float(I.mu)/I.L
    
    # Action space:
    # If j = 0, but there are spare ALS units, then...
    # Action 0: Redirect next BLS call
    # Action 1: Route next BLS call to ALS unit
    # Otherwise, a dummy action.
    
    A = {} # create empty action matrix to fill in the loop below
    # loop through all possible states
    for s in S:
        i, j = I.lookup[s]
        # if we are in the boundary condition, then assign two options to the action
        if i < I.NA and j == I.NB:
            A[s] = [0, 1] # take action 0 or 1
        else:
            A[s] = [0] # action 0 the only option
    
    # Reward matrix initially filled with zero.
    R = np.zeros((N, 2))
    # loop through each state again and assign the reward based on the actions and state
    for s in S:
        i, j = I.lookup[s]
        if i < I.NA:
            if j < I.NB:
                # if both ALS and BLS units are not at capacity then the reward is simply the probability of a high priority
                # call arriving times the reward for servicing the high priority call plus the probability of a low priority call
                # times the reward for servicing a low priority call
                R[s,:] = lh*I.RHA + ll*I.RL
            else:
                # if there are spare ALS (i < I.NB) but all BLS are busy (j >= to I.NB),
                # then the reward for taking action zero (redirecting the call, ignoring BLS) is the probability of a high priority call times
                # the reward for servicing a high priority call
                R[s,0] = lh*I.RHA
                # If you take action 1 (sending an ALS to a low priority call), then the reward is the same as above plus
                # the probability of a low priority call times the reward for servicing a low priority call.
                R[s,1] = lh*I.RHA + ll*I.RL
        elif j < I.NB:
            # if ALS is at capacity and there are available BLS vehicles, the the reward function is the probability of high priorty call
            # times the reward for a BLS servicing a high priority call plus probability of a low priority call times the reward for
            # servicing a low priority call.
            R[s,:] = lh*I.RHB + ll*I.RL
    
    # Transition probability matrices - two matrices of size 174*174
    # P[a][i][j] - One-step transition prob. from states i to j under action a
    P = np.zeros((2,N,N)) # P is a list of two matrices. The matrices are equivalent except for states that require an action
    #

    # loop through all states
    for s in S:
        # get the number of active ALS and BLS for each state
        i, j = I.lookup[s]

        # generally just 0, but 0 and 1 for state spaces with two possible actions. If zero, it filles
        # the probability matrix for action 0, if 1 it filles the probability matrix for action 1
        for a in A[s]:
            # ALS arrivals (if ALS has idle vehicles, then assign the probability of a high priority call
            if i < I.NA:
                P[a][s][I.revlookup[(i+1, j)]] += lh # probability of adding an ALS
            # if ALS are all busy, but BLS has capacity, then
            elif j < I.NB:
                P[a][s][I.revlookup[(i, j+1)]] += lh

            # BLS arrival (if BLS has idle vehicles)
            if j < I.NB:
                P[a][s][I.revlookup[(i, j+1)]] += ll # probability of adding a BLS
                # a is 0 or 1 which python can interpret as a true false (1=true). so in this case
                # all BLS are busy and but we have ALS capacity and a is true (meaning we take action 1)
            elif i < I.NA and a:
                P[a][s][I.revlookup[(i+1, j)]] += ll # probability of adding a BLS

            # ALS service completion
            if i > 0:
                P[a][s][I.revlookup[(i-1, j)]] += i*mu # probability of ALS being freed up

            # BLS service completion
            if j > 0:
                P[a][s][I.revlookup[(i, j-1)]] += j*mu # probability of BLS being freed up
    
            # Dummy transitions
            P[a][s][I.revlookup[(i, j)]] = 1 - sum([P[a][s][t] for t in S])

    return S, R, A, P


# this function takes the Instance class and creates dictionaries for all possible combinations of ALS and BLS to
# iterate through.
def build_dicts(I):
    # State space: (i, j)
    # i = Busy ALS units
    # j = Busy BLS units
    
    lookup = {}
    revlookup = {}
    count = 0
    
    for i in range(I.NA + 1):
        for j in range(I.NB + 1):
                    lookup[count] = (i, j)
                    revlookup[(i,j)] = count
                    count += 1
                    
    return lookup, revlookup


