from assorted_mdp import *
import base_model
import policy_iteration
import numpy as np
from matplotlib       import pyplot as plt

class instance:
    pass

def create_output(I, fileName):
    # Set of feasible fleets
    maxALS = int(I.budget/I.CA)
    fleets = [(i, int((I.budget - i*I.CA)/I.CB)) for i in xrange(maxALS + 1)]
    xaxis  = range(maxALS + 1)

    
    # Performance of each vehicle mix: the y-axis
    yaxis  = [0]*len(fleets)

    # Input parameters pA, pB, and f of the linear program obtained from
    #   the stationary distribution of the optimal policy
    #   Not used in the computational study for the MDP.
    LPparams = [[0, 0, 0]]*len(fleets)
    
    # Obtaining plot of performance with respect to vehicle mix
    for i in xrange(len(fleets)):
        I.NA = fleets[i][0]
        I.NB = fleets[i][1]
        I.L  = I.lamH + I.lamL + I.mu*(I.NA + I.NB)

        S, R, A, P     = base_model.build_mdp(I, False)
        J, _, u, P, _  = policy_iteration.solve_mdp(S, R, A, P)
        LPparams[i]    = getLPparams(I, P, S, A, u)
        yaxis[i]       = I.L*J

    # Writing output to file
    with open(fileName, 'w') as f:
        f.write('%i\n' % len(fleets))
        for i in xrange(len(fleets)):
            f.write('%i %i %.8f \n' % (fleets[i][0], fleets[i][1], yaxis[i]))

    # Writing IP input parameters to file
    with open('LPparams_basecase.txt', 'w') as f:
        f.write('%i\n' % len(fleets))
        for i in xrange(len(fleets)):
            f.write('%i %i %.8f %.8f %.8f \n' % (fleets[i][0], fleets[i][1],\
                     LPparams[i][0], LPparams[i][1], LPparams[i][2]))

    return xaxis, yaxis, fleets

def read_output(fileName):
    with open(fileName, 'r') as f:
        numFleets = int(f.readline())
        xaxis     = [0]*numFleets
        yaxis     = [0]*numFleets
        fleets    = [0]*numFleets
        
        for i in xrange(numFleets):
            line      = f.readline().split()
            fleets[i] = (int(line[0]), int(line[1]))
            xaxis[i]  = int(line[0])
            yaxis[i]  = float(line[2])
			
    return xaxis, yaxis, fleets


I       = instance()
I.RHA   = 1
I.RHB   = 0.5
I.RL    = 0.6
I.lamH  = 8
I.lamL  = 13
I.mu    = 0.75
I.CA    = 1.25
I.CB    = 1
high    = 1  # Utilization or service level based budget?

if high:
    # Determining the EMS's operating budget, which in this case, is 50
    #   (40 ALS units to bring utilization < 0.7)
    # I.budget = assorted_mdp.find_budget(I, 0.7)
    I.budget = 50.0
    xaxis, yaxis, fleets = create_output(I, 'basecase_output.txt')
    xaxis, yaxis, fleets = read_output('basecase_output.txt')
else:
    # If we go instead by service, level, 34 ALS units to reach > 95% of calls
    # I.budget = assorted_mdp.find_budget_alternate(I, 0.95)
    I.budget = 42.5
    xaxis, yaxis, fleets = create_output(I, 'basecase2_output.txt')
    xaxis, yaxis, fleets = read_output('basecase2_output.txt')


# Mixture plot
plt.plot(xaxis, yaxis, linewidth = 2)
plt.xlabel('ALS Fleet Size', fontsize = 20)
plt.ylabel('Long-Run Avg. Reward', fontsize = 20)
plt.xticks(np.arange(5,45,5), fontsize = 16)
plt.yticks(np.arange(2,18,2), fontsize = 16)
plt.xlim([0, fleets[-1][0]])
plt.ylim([0, int(max(yaxis)) + 1])
plt.legend(loc = 4, prop={'size':16})
plt.grid(linewidth = 1)

if high:
    plt.savefig('basecase.pdf')
else:
    plt.savefig('blah.pdf')
    #plt.savefig('basecase2.pdf')
    
plt.show()
plt.close('all')




