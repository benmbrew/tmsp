from assorted_mdp import * # import all the functions created in the assorted_mdp.py file
import base_model # import base model script
import policy_iteration # import policy iteration script
import numpy as np # import numpy package
from matplotlib import pyplot as plt # import pyplot from matplotlib package with alias plt (

# --------------------------------
### This script is the main file to produce the first plot in Chong Vehicle mix paper
# --------------------------------

# create a class called instance (this object will hold all the information about inputs to the MDP - arrival rates, completion rates, budget, etc)
class instance:
    pass

# This function takes the instance and name of the file, builds the mdp, does policy iteration and writes the output to the current directory

def create_output(I, fileName):
    # divide buget by ALS cost to get maximum number of ALS vehicles
    maxALS = int(I.budget / I.CA)
    # create all the different fleet possibilities with a given budget
    fleets = [(i, int((I.budget - i * I.CA) / I.CB)) for i in range(maxALS + 1)]
    # define the x axis range (0-70 in the base case)
    xaxis = range(maxALS + 1)

    # Define the y axis (a list of length 71). filled with zero
    yaxis = [0] * len(fleets)

    # Input parameters pA, pB, and f of the linear program obtained from
    # the stationary distribution of the optimal policy
    # Not used in the computational study for the MDP.
    LPparams = [[0, 0, 0]] * len(fleets)

    # Obtaining plot of performance with respect to vehicle mix
    # loops through 0 to 71. It essentially iterates through all combinations of ALS and BLS, based on budget and maxALS.
    # for example, when i=0, it evaluates the case where ALS =0, and BLS = 87
    for i in range(len(fleets)):
        I.NA = fleets[i][0] #
        I.NB = fleets[i][1]
        # get the maximal rate - the denominator for uniformization for probability matrix
        I.L = I.lamH + I.lamL + I.mu * (I.NA + I.NB)

        # Build the mdp. Based on the parameters in the I class, create the matrices
        # S = State space matrix
        # R = Reward matrix
        # A = Action matrix
        # P = Probability transition matrix
        S, R, A, P = base_model.build_mdp(I, False)
        # run the policy iteration to get long run average reward
        J, _, u, P, _ = policy_iteration.solve_mdp(S, R, A, P)
        # the code below is not used in mdp but we removed the S from getLPparams in order for it to run (only takes 4 arguments, not 5).
        LPparams[i] = getLPparams(I, P, A, u)
        # add reward to y axis
        yaxis[i] = I.L * J

    # Writing output to file
    with open(fileName, 'w') as f:
        f.write('%i\n' % len(fleets))
        for i in range(len(fleets)):
            f.write('%i %i %.8f \n' % (fleets[i][0], fleets[i][1], yaxis[i]))

    # Writing IP input parameters to file
    with open('LPparams_basecase.txt', 'w') as f:
        f.write('%i\n' % len(fleets))
        for i in range(len(fleets)):
            f.write('%i %i %.8f %.8f %.8f \n' % (fleets[i][0], fleets[i][1], \
                                                 LPparams[i][0], LPparams[i][1], LPparams[i][2]))

    return xaxis, yaxis, fleets


# read in the results
def read_output(fileName):
    with open(fileName, 'r') as f: # open file (call it "f")
        numFleets = int(f.readline()) # gets total number of lines (number of combinations to evaluate)
        xaxis = [0] * numFleets # fill with zero
        yaxis = [0] * numFleets # fill with zero
        fleets = [0] * numFleets # fill with zero

        # loop through all mdp instances
        for i in range(numFleets):
            line = f.readline().split()
            fleets[i] = (int(line[0]), int(line[1])) # extract number of ALS and BLS
            xaxis[i] = int(line[0]) # get ALS for x axis
            yaxis[i] = float(line[2]) # get reward associated with number of ALS
    return xaxis, yaxis, fleets # return the axes and fleets for plotting


# inputs to the MDP --------------
I = instance()
I.RHA = 1 # reward for an ALS completing a high priority call
I.RHB = 0.5 # reward for an BLS completing a low priority call
I.RL = 0.6 # reward for dispatching ALS or BLS to low priority call
I.lamH = 8.1 # arrival rate (per hour) of high priority calls
I.lamL = 13.1 # arrival rate (per hour) of low priority calls
I.mu = 0.75 # service completion rate (hourly)
I.CA = 1.25 # cost of an ALS vehicle (as a ratio to BLS)
I.CB = 1 # cost of BLS vehicle
I.budget = 87.5 # Budget

# create output and read it in.
xaxis, yaxis, fleets = create_output(I, 'basecase_output.txt')
xaxis, yaxis, fleets = read_output('basecase_output.txt')

# Mixture plot (from first figure)
plt.plot(xaxis, yaxis, linewidth=2) # create basic line plot with number of ALS on x axis and corresponding long run avg reward on y axis
plt.xlabel('ALS Fleet Size', fontsize=15) # add x axis label
plt.ylabel('Long-Run Avg. Reward', fontsize=15) # add y axis label
plt.xticks(np.arange(5, 70, 5), fontsize=12) # add x axis tick marks
plt.yticks(np.arange(10, 18, 2), fontsize=12) # add y axis tick marks
plt.xlim([0, fleets[-1][0]]) # add x axis limit
plt.ylim([10, int(max(yaxis)) + 1]) # add y axis limit
plt.legend(loc=4, prop={'size': 12}) # add a legend to the "4th location"
plt.grid(linewidth=1) # add grid to plot
plt.show() # show plot (depending on your version of matplotlib, this might not work or be necessary.
plt.close('all') # close plot




