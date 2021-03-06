import numpy as np

def stationaryDist(S, P, u):
    # Given a policy u, and the transition matrices of a given state-action
    #   pair, computes the stationary distribution of the induced Markov chain.
    M = np.empty((len(S), len(S)))
    for s in S:
        M[s,:] = P[u[s]][s]

    z = len(S) - 1
    Q = np.eye(len(S))- M.T
    Q[z,:] = 1
    rhs = np.zeros((len(S),1))
    rhs[z] = 1

    '''
    # Long-run average cost associated with policy uold
    # We run into numerical stability issues when attempting to solve
    #   linear stystems here. Naive analysis suggests that systems
    #   with a determinant that Python determines to be a very small
    #   negative number cause trouble. Selecting a "distinguished state"
    #   in the "middle" of the state space seems to alleviate this problem.
        

        
    if np.linalg.det(Q) <= 1e-25:
        z      = 0
        Q      = np.eye(N)- Pu.T
        Q[z,:] = 1
        
        if np.linalg.det(Q) <= 1e-25:
            z      = -1
            Q      = np.eye(N)- Pu.T
            Q[z,:] = 1
                    
    rhs    = np.zeros((N,1))
    rhs[z] = 1
    '''
    return np.linalg.solve(Q, rhs)

def relativeVF(S, P, J, u, r):
    M = np.empty((len(S), len(S)))
    z = len(S) - 1
    for s in S:
        M[s,:] = P[u[s]][s]
        
    Q = np.eye(len(S)) - M
    Q[z,:] = 0
    Q[z,z] = 1

    '''
    if np.linalg.det(Q) <= 1e-25:
        z      = 0
        Q      = np.eye(N) - Pu
        Q[z,:] = 0
        Q[z,z] = 1
        
        if np.linalg.det(Q) <= 1e-25:
            z      = -1
            Q      = np.eye(N) - Pu
            Q[z,:] = 0
            Q[z,z] = 1
    '''
           
    rhs = r - J*np.ones((len(S),1))
    rhs[z] = 0
    return np.linalg.solve(Q, rhs)
    
    
#this function takes the State, Reward, Action, and Probability matrices and return the reward

def solve_mdp(S, R, A, P):
    # Policy iteration. Outputs the optimal policy, as well as the associated
    #   value function.
    # Assumed that state space of the form S = {0, 1, ..., N-1}
    # u denotes control, pi stationary distributions
    
    N = len(S) # get number of states
    uold = [-1]*N # control initially same length as state, filled with -1
    unew = [A[s][0] for s in S] # gets only the 0 action
    r = np.zeros((N, 1))
    iters = 0

    # In some cases, policies oscillate, due to floating point precision
    #   concerns. Alternate termination criterion: if the average cost
    #   associated with policies in two consecutive iterations differ by
    #   less than a prespecified threshold epsilon
    Jold = 0
    Jnew = 1

    # while this condition is true, run the following.
    # Keep running the iteration until its stable uold=unew and Jnew~Jold
    while uold != unew and abs(Jnew - Jold) > 1e-7:
        uold = list(unew)
        Jold = Jnew
        
        # Trans. matrix, reward vector
        # for each state get the reward based on the uniformized probabilities
        for s in S:
            r[s] = R[s,uold[s]] # get the reward for state and action

        pi = stationaryDist(S, P, uold) # get the stationary distribution
        Jnew = np.dot(r.T,pi) # get the dot product of the reward vector and the stationary distribution
        h = relativeVF(S, P, Jnew, uold, r) # relative value function

        # Solving optimality equations for updated policy
        # for each state, calculate the value function for each action
        for s in S:
            bestv = -9999

            for a in A[s]:
                # matrix for each state and action
                v = R[s,a] + np.dot(P[a][s], h) # the value function.
                if v > bestv: # determines which action is better
                    bestv = v
                    unew[s] = a

        iters += 1

    # Solving for the stationary distribution of the terminating policy
    #   (may not be the same as pi as calced above, due to floating point
    #     precision errors)
    pi = stationaryDist(S, P, unew)

    return float(Jnew), h, unew, pi, iters

'''
# Debugging with a simple example
S = [0, 1]
A = [[0, 1], [0, 1]]
R = np.mat([[0, 1], [1, 2]])
P = np.zeros((2,2,2))
P[0][0][0] = 1/3.
P[0][0][1] = 2/3.
P[0][1][0] = 1/2.
P[0][1][1] = 1/2.
P[1][0][0] = 1/3.
P[1][0][1] = 2/3.
P[1][1][0] = 1/2.
P[1][1][1] = 1/2.

J, h, _, _, _ = solve_mdp(S, R, A, P)
print J
print h
'''
