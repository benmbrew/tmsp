import numpy as np
from mdptoolbox import mdp

# The number of population abundance classes
POPULATION_CLASSES = 7
# The number of years since a fire classes
FIRE_CLASSES = 13
# The number of states
STATES = POPULATION_CLASSES * FIRE_CLASSES
# The number of actions
ACTIONS = 2
ACTION_NOTHING = 0
ACTION_BURN = 1

def check_action(x):
    """Check that the action is in the valid range."""
    if not (0 <= x < ACTIONS):
        msg = "Invalid action '%s', it should be in {0, 1}." % str(x)
        raise ValueError(msg)

def check_population_class(x):
    """Check that the population abundance class is in the valid range."""
    if not (0 <= x < POPULATION_CLASSES):
        msg = "Invalid population class '%s', it should be in {0, 1, …, %d}." \
              % (str(x), POPULATION_CLASSES - 1)
        raise ValueError(msg)

def check_fire_class(x):
    """Check that the time in years since last fire is in the valid range."""
    if not (0 <= x < FIRE_CLASSES):
        msg = "Invalid fire class '%s', it should be in {0, 1, …, %d}." % \
              (str(x), FIRE_CLASSES - 1)
        raise ValueError(msg)

def check_probability(x, name="probability"):
    """Check that a probability is between 0 and 1."""
    if not (0 <= x <= 1):
        msg = "Invalid %s '%s', it must be in [0, 1]." % (name, str(x))
        raise ValueError(msg)


def get_habitat_suitability(years):
    """The habitat suitability of a patch relatve to the time since last fire.

    The habitat quality is low immediately after a fire, rises rapidly until
    five years after a fire, and declines once the habitat is mature. See
    Figure 2 in Possingham and Tuck (1997) for more details.

    Parameters
    ----------
    years : int
        The time in years since last fire.

    Returns
    -------
    r : float
        The habitat suitability.

    """
    if years < 0:
        msg = "Invalid years '%s', it should be positive." % str(years)
        raise ValueError(msg)
    if years <= 5:
        return 0.2*years
    elif 5 <= years <= 10:
        return -0.1*years + 1.5
    else:
        return 0.5



def convert_state_to_index(population, fire):
    """Convert state parameters to transition probability matrix index.

    Parameters
    ----------
    population : int
        The population abundance class of the threatened species.
    fire : int
        The time in years since last fire.

    Returns
    -------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.

    """
    check_population_class(population)
    check_fire_class(fire)
    return population*FIRE_CLASSES + fire


def convert_index_to_state(index):
    """Convert transition probability matrix index to state parameters.

    Parameters
    ----------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.

    Returns
    -------
    population, fire : tuple of int
        ``population``, the population abundance class of the threatened
        species. ``fire``, the time in years since last fire.

    """
    if not (0 <= index < STATES):
        msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
              (str(index), STATES - 1)
        raise ValueError(msg)
    population = index // FIRE_CLASSES
    fire = index % FIRE_CLASSES
    return (population, fire)


prob = np.zeros(STATES)
r = get_habitat_suitability(F)


def transition_fire_state(F, a):
    """Transition the years since last fire based on the action taken.

    Parameters
    ----------
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    F : int
        The time in years since last fire.

    """
    ## Efect of action on time in years since fire.
    if a == ACTION_NOTHING:
        # Increase the time since the patch has been burned by one year.
        # The years since fire in patch is absorbed into the last class
        if F < FIRE_CLASSES - 1:
            F += 1
    elif a == ACTION_BURN:
        # When the patch is burned set the years since fire to 0.
        F = 0

    return F


def get_transition_probabilities(s, x, F, a):
    """Calculate the transition probabilities for the given state and action.

    Parameters
    ----------
    s : float
        The class-independent probability of the population staying in its
        current population abundance class.
    x : int
        The population abundance class of the threatened species.
    F : int
        The time in years since last fire.
    a : int
        The action undertaken.

    Returns
    -------
    prob : array
        The transition probabilities as a vector from state (``x``, ``F``) to
        every other state given that action ``a`` is taken.

    """
    # Check that input is in range
    check_probability(s)
    check_population_class(x)
    check_fire_class(F)
    check_action(a)

    # a vector to store the transition probabilities
    prob = np.zeros(STATES)

    # the habitat suitability value
    r = get_habitat_suitability(F)
    F = transition_fire_state(F, a)

    ## Population transitions
    if x == 0:
        # population abundance class stays at 0 (extinct)
        new_state = convert_state_to_index(0, F)
        prob[new_state] = 1
    elif x == POPULATION_CLASSES - 1:
        # Population abundance class either stays at maximum or transitions
        # down
        transition_same = x
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == ACTION_BURN:
            transition_same -= 1
            transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = 1 - (1 - s)*(1 - r)
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        prob[new_state] = (1 - s)*(1 - r)
    else:
        # Population abundance class can stay the same, transition up, or
        # transition down.
        transition_same = x
        transition_up = x + 1
        transition_down = x - 1
        # If action 1 is taken, then the patch is burned so the population
        # abundance moves down a class.
        if a == ACTION_BURN:
            transition_same -= 1
            transition_up -= 1
            # Ensure that the abundance class doesn't go to -1
            if transition_down > 0:
                transition_down -= 1
        # transition probability that abundance stays the same
        new_state = convert_state_to_index(transition_same, F)
        prob[new_state] = s
        # transition probability that abundance goes up
        new_state = convert_state_to_index(transition_up, F)
        prob[new_state] = (1 - s)*r
        # transition probability that abundance goes down
        new_state = convert_state_to_index(transition_down, F)
        # In the case when transition_down = 0 before the effect of an action
        # is applied, then the final state is going to be the same as that for
        # transition_same, so we need to add the probabilities together.
        prob[new_state] += (1 - s)*(1 - r)

    # Make sure that the probabilities sum to one
    assert (prob.sum() - 1) < np.spacing(1)
    return prob


def get_transition_and_reward_arrays(s):
    """Generate the fire management transition and reward matrices.

    The output arrays from this function are valid input to the mdptoolbox.mdp
    classes.

    Let ``S`` = number of states, and ``A`` = number of actions.

    Parameters
    ----------
    s : float
        The class-independent probability of the population staying in its
        current population abundance class.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrices P and
        ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
        numpy array and R is a numpy vector of length ``S``.

    """
    check_probability(s)

    # The transition probability array
    transition = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    reward = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Get the state index as inputs to our functions
        x, F = convert_index_to_state(idx)
        # The reward for being in this state is 1 if the population is extant
        if x != 0:
            reward[idx] = 1
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            transition[a][idx] = get_transition_probabilities(s, x, F, a)

    return (transition, reward)


def solve_mdp():
    """Solve the problem as a finite horizon Markov decision process.

    The optimal policy at each stage is found using backwards induction.
    Possingham and Tuck report strategies for a 50 year time horizon, so the
    number of stages for the finite horizon algorithm is set to 50. There is no
    discount factor reported, so we set it to 0.96 rather arbitrarily.

    Returns
    -------
    sdp : mdptoolbox.mdp.FiniteHorizon
        The PyMDPtoolbox object that represents a finite horizon MDP. The
        optimal policy for each stage is accessed with mdp.policy, which is a
        numpy array with 50 columns (one for each stage).

    """
    P, R = get_transition_and_reward_arrays(0.5)
    sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
    sdp.run()
    return sdp


def print_policy(policy):
    """Print out a policy vector as a table to console

    Let ``S`` = number of states.

    The output is a table that has the population class as rows, and the years
    since a fire as the columns. The items in the table are the optimal action
    for that population class and years since fire combination.

    Parameters
    ----------
    p : array
        ``p`` is a numpy array of length ``S``.

    """
    p = np.array(policy).reshape(POPULATION_CLASSES, FIRE_CLASSES)
    print("    " + " ".join("%2d" % f for f in range(FIRE_CLASSES)))
    print("    " + "---" * FIRE_CLASSES)
    for x in range(POPULATION_CLASSES):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                     range(FIRE_CLASSES)))


sdp = solve_mdp()
print_policy(sdp.policy[:, 0])