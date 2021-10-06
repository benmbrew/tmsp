import numpy as np
# No buffer or queue

class instance:
    pass

def build_mdp(I, debug):
    # Create dicionaries to lookup / reverse lookup elements in state space
    I.lookup, I.revlookup = build_dicts(I)
    N = len(I.lookup)
    S = range(N)

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
    
    A = {}
    for s in S:
        i, j = I.lookup[s]
        # if we are in the boundary condition, then assign two options to the action
        if i < I.NA and j == I.NB:
            A[s] = [0, 1]
        else:
            A[s] = [0]
    
    # Reward matrix
    R = np.zeros((N, 2))
    for s in S:
        i, j = I.lookup[s]
        if i < I.NA:
            if j < I.NB:
                R[s,:] = lh*I.RHA + ll*I.RL
            else:
                R[s,0] = lh*I.RHA 
                R[s,1] = lh*I.RHA + ll*I.RL
        elif j < I.NB:
            R[s,:] = lh*I.RHB + ll*I.RL
    
    # Transition probability matrices - two matrices of size 174*174
    # P[a][i][j] - One-step transition prob. from states i to j under action a
    P = np.zeros((2,N,N))

    for s in S:
        i, j = I.lookup[s]

        for a in A[s]:
            # ALS arrivals
            if i < I.NA:
                P[a][s][I.revlookup[(i+1, j)]] += lh
            elif j < I.NB:
                P[a][s][I.revlookup[(i, j+1)]] += lh

            # BLS arrival
            if j < I.NB:
                P[a][s][I.revlookup[(i, j+1)]] += ll
            elif i < I.NA and a:
                P[a][s][I.revlookup[(i+1, j)]] += ll

            # ALS service completion
            if i > 0:
                P[a][s][I.revlookup[(i-1, j)]] += i*mu                

            # BLS service completion
            if j > 0:
                P[a][s][I.revlookup[(i, j-1)]] += j*mu
    
            # Dummy transitions
            P[a][s][I.revlookup[(i, j)]] = 1 - sum([P[a][s][t] for t in S])
            
    return S, R, A, P


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


