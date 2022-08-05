# Implementation of Game Theory Models for Pursuit Evasion Games by Khan
# https://www.cs.ubc.ca/~kevinlb/teaching/cs532a%20-%202006-7/projects/EMTreport.pdf

import math
import numpy as np
from matplotlib import pyplot as plt

# dimensions and limits
WORLD_X = 40
WORLD_Y = 40
MAX_TIME = 2000
NUM_RUNS = 100
EPISODE_LIMIT = 1000
FINAL_REWARD = MAX_TIME * WORLD_X * math.sqrt(2)

# possible actions of the agents
STAY = 0
NORTH = 1  # 0
EAST = 2   # 90
SOUTH = 3  # 180
WEST = 4   # 270
NE = 5     # 45
SE = 6     # 135
SW = 7     # 225
NW = 8     # 315
EVADER_ACTIONS = [STAY, NORTH, EAST, SOUTH, WEST, NE, SE, SW, NW]
EVADER_ACTION_SPACE_SIZE = len(EVADER_ACTIONS)
PURSUER_ACTIONS = [STAY, NORTH, EAST, SOUTH, WEST]
PURSUER_ACTION_SPACE_SIZE = len(PURSUER_ACTIONS)

# initial state
PURSUER_START = [0, 0]
EVADER_START = [WORLD_X - 1, WORLD_Y - 1]
ESCAPE_SQ1 = [WORLD_X - 1, 0]
ESCAPE_SQ2 = [0, WORLD_Y - 1]
INIT_STATE = [PURSUER_START, EVADER_START]

# results
PURSUER_WIN = 1
EVADER_WIN = -1
DRAW = 0


# This function defines how the pursuer and evader move on the grid.
# gremlin represents the probability that strategically selected action is replaced with a random action
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, pursuer_action, evader_action, gremlin):

    pursuer_pos, evader_pos = state
    pursuer_x = pursuer_pos[0]
    pursuer_y = pursuer_pos[1]
    evader_x = evader_pos[0]
    evader_y = evader_pos[1]
    dx = 0
    dy = 0

    # pursuer action

    if np.random.binomial(1, gremlin) == 1:
        pursuer_action = np.random.choice(PURSUER_ACTIONS)

    if pursuer_action == NORTH:
        pursuer_pos = [max(min(pursuer_x - 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + dy,     WORLD_Y - 1), 0)]
    elif pursuer_action == EAST:
        pursuer_pos = [max(min(pursuer_x + dx,     WORLD_X - 1), 0), max(min(pursuer_y + 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == SOUTH:
        pursuer_pos = [max(min(pursuer_x + 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + dy,     WORLD_Y - 1), 0)]
    elif pursuer_action == WEST:
        pursuer_pos = [max(min(pursuer_x + dx,     WORLD_X - 1), 0), max(min(pursuer_y - 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == STAY:
        pursuer_pos = [max(min(pursuer_x + dx,     WORLD_X - 1), 0), max(min(pursuer_y + dy,     WORLD_Y - 1), 0)]

    # evader action

    if np.random.binomial(1, gremlin) == 1:
        evader_action = np.random.choice(EVADER_ACTIONS)

    if evader_action == NORTH:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y + dy,     WORLD_Y - 1), 0)]
    elif evader_action == NE:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == EAST:
        evader_pos = [max(min(evader_x + dx,     WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == SE:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == SOUTH:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y + dy,     WORLD_Y - 1), 0)]
    elif evader_action == SW:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == WEST:
        evader_pos = [max(min(evader_x + dx,     WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == NW:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == STAY:
        evader_pos = [max(min(evader_x + dx,     WORLD_X - 1), 0), max(min(evader_y + dy,     WORLD_Y - 1), 0)]

    state = [pursuer_pos, evader_pos]
    return state


# play for an episode, return amount of time spent in the episode
# game keeps going until (1) evader reaches one of the escape squares or (2) pursuer catches evader
def episode(pursuer_q_value, evader_q_value, eps, gremlin, alpha, plot):

    # initialize the counter that will track the total time taken for this episode
    t = 0

    # initialize state
    state = INIT_STATE
    pursuer_pos = PURSUER_START
    evader_pos = EVADER_START
    pursuer_wins = False
    evader_wins = False
    draw = False
    result = ""

    # choose initial pursuer and evader actions randomly
    pursuer_action = np.random.choice(PURSUER_ACTIONS)
    evader_action = np.random.choice(EVADER_ACTIONS)

    while (not pursuer_wins) and (not evader_wins) and (not draw):

        # determine the next state
        next_state = step(state, pursuer_action, evader_action, gremlin)

        # choose the next pursuer action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:

            # eps% of the time, select action randomly
            next_pursuer_action = np.random.choice(PURSUER_ACTIONS)

        else:

            # (1-eps)% of the time, we select an action greedily
            # Select the action associated with the maximum q_value found among all q_values stored in values_
            # for each action_ and value_ stored in values_
            #   if the variable value_ is maximum in the list values_
            #     select the action_ associated with that q_value.
            pursuer_values_ = pursuer_q_value[pursuer_pos[0], pursuer_pos[1], :]
            next_pursuer_action = np.random.choice(
                [action_ for action_, value_ in enumerate(pursuer_values_) if (value_ == np.max(pursuer_values_)).any()
                 ])

        # choose the next evader action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:
            next_evader_action = np.random.choice(EVADER_ACTIONS)
        else:
            evader_values_ = pursuer_q_value[evader_pos[0], evader_pos[1], :]
            next_evader_action = np.random.choice(
                [action_ for action_, value_ in enumerate(evader_values_) if (value_ == np.max(evader_values_)).any()
                 ])

        # set up separate reward structures for pursuer and evader
        if pursuer_pos == evader_pos:
            pursuer_reward = FINAL_REWARD
        else:
            pursuer_reward = -distance(evader_pos, pursuer_pos)
        if (evader_pos == ESCAPE_SQ1) or (evader_pos == ESCAPE_SQ2):
            evader_reward = FINAL_REWARD
        else:
            evader_reward = -min(distance(evader_pos, ESCAPE_SQ1), distance(evader_pos, ESCAPE_SQ2))

        # SARSA update - For more info about SARSA algorithm, please refer to p. 129 of the S&B textbook.

        # improved reward for terminal state
        pursuer_q_value[pursuer_pos[0], pursuer_pos[1], pursuer_action] = \
            pursuer_q_value[pursuer_pos[0], pursuer_pos[1], next_pursuer_action] \
            + alpha * (pursuer_reward + pursuer_q_value[pursuer_pos[0], pursuer_pos[1], next_pursuer_action]
                       - pursuer_q_value[pursuer_pos[0], pursuer_pos[1], next_pursuer_action])
        evader_q_value[evader_pos[0], evader_pos[1], evader_action] = \
            evader_q_value[evader_pos[0], evader_pos[1], next_evader_action] \
            + alpha * (evader_reward + evader_q_value[evader_pos[0], evader_pos[1], next_evader_action]
                       - evader_q_value[evader_pos[0], evader_pos[1], next_evader_action])

        # if this is the last episode of the last run (i.e., max learning has occurred), save plots for animation
        if plot:
            circle1 = plt.Circle((pursuer_pos[0], pursuer_pos[1]), 0.25, color='r')
            circle2 = plt.Circle((evader_pos[0], evader_pos[1]), 0.25, color='g')
            fig, ax = plt.subplots()
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            plt.xlim(0, WORLD_X)
            plt.ylim(0, WORLD_Y)
            plt.title('Game Board')
            plt.grid()
            filename = "plt" + str(t) + ".png"
            fig.savefig(filename)
            plt.close(fig)

        # update variables
        state = next_state
        pursuer_action = next_pursuer_action
        evader_action = next_evader_action
        pursuer_pos, evader_pos = state
        t += 1

        # determine game state
        pursuer_wins = (pursuer_pos == evader_pos)
        evader_wins = (evader_pos == ESCAPE_SQ1) or (evader_pos == ESCAPE_SQ2)
        draw = (t == MAX_TIME)
        if pursuer_wins:
            result = PURSUER_WIN
        if evader_wins:
            result = EVADER_WIN
        if draw:
            result = DRAW

    return t, state, result


def runner(eps, gremlin, alpha, pursuer_q_value, evader_q_value, last_run):

    pursuer_win_cnt = 0
    evader_win_cnt = 0

    ep = 0
    while ep < EPISODE_LIMIT:
        ep += 1
        last_episode = (ep == EPISODE_LIMIT)
        t, s, r = episode(pursuer_q_value, evader_q_value, eps, gremlin, alpha, last_run and last_episode)
        if r == PURSUER_WIN:
            pursuer_win_cnt += 1
        elif r == EVADER_WIN:
            evader_win_cnt += 1
        # ignore draws

    pursuer_win_pct = pursuer_win_cnt / EPISODE_LIMIT
    evader_win_pct = evader_win_cnt / EPISODE_LIMIT
    pursuer_win_pct = round((pursuer_win_pct / (pursuer_win_pct + evader_win_pct))*100, 1)  # ignore draws
    return pursuer_win_pct


def distance(square1, square2):
    x_d = (square1[0] - square2[0])
    y_d = (square1[1] - square2[1])**2
    return math.sqrt(x_d**2 + y_d**2)


def simulator(eps, gremlin, alpha):

    # allow the Q values to improve from one run to the next
    pursuer_q_value = np.zeros((WORLD_X, WORLD_Y, PURSUER_ACTION_SPACE_SIZE))
    evader_q_value = np.zeros((WORLD_X, WORLD_Y, EVADER_ACTION_SPACE_SIZE))

    run = 0
    agg_sum = 0
    agg_sq_sum = 0

    while run < NUM_RUNS:
        last_run = (run == NUM_RUNS - 1)
        res = runner(eps, gremlin, alpha, pursuer_q_value, evader_q_value, last_run)
        agg_sum += res
        agg_sq_sum += res**2
        run += 1
        print("runs: ", run, ", pursuer win pct: ", round(agg_sum/run, 1), "%")

    avg = agg_sum / NUM_RUNS
    avg_sq = agg_sq_sum / NUM_RUNS
    std_err = math.sqrt(avg_sq - avg**2) / math.sqrt(NUM_RUNS)
    ret_str = "Pursuer Win Pct = " + str(round(avg, 1)) + "% with standard error " + str(round(std_err, 1)) + "%"
    return ret_str


if __name__ == '__main__':

    print("Baseline scenario: ", simulator(0.2, 0.1, 0.5))
    # print("Increase alpha by 20%: ", simulator(0.2, 0.1, 0.6))
    # print("Decrease alpha by 20%: ", simulator(0.2, 0.1, 0.4))
    # print("Increase epsilon by 20%: ", simulator(0.3, 0.1, 0.5))
    # print("Decrease epsilon by 20%: ", simulator(0.3, 0.1, 0.5))
    # print("Double the noise: ", simulator(0.2, 0.2, 0.5))
    # print("Half the noise: ", simulator(0.2, 0.05, 0.5))
