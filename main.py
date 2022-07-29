import numpy as np
import time

# dimensions
WORLD_X = 20
WORLD_Y = 20
MAX_TIME = 2500

# possible actions of the robot
NORTH = 0  # 0
NE = 1  # 45
EAST = 2  # 90
SE = 3  # 135
SOUTH = 4  # 180
SW = 5  # 225
WEST = 6  # 270
NW = 7  # 315
ACTIONS = [NORTH, NE, EAST, SE, SOUTH, SW, WEST, NW]
ACTION_SPACE_SIZE = 8

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
        pursuer_action = np.random.choice(ACTIONS)

    if pursuer_action == NORTH:
        pursuer_pos = [max(min(pursuer_x - 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + dy,     WORLD_Y - 1), 0)]
    elif pursuer_action == NE:
        pursuer_pos = [max(min(pursuer_x - 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == EAST:
        pursuer_pos = [max(min(pursuer_x + dx,     WORLD_X - 1), 0), max(min(pursuer_y + 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == SE:
        pursuer_pos = [max(min(pursuer_x + 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == SOUTH:
        pursuer_pos = [max(min(pursuer_x + 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y + dy,     WORLD_Y - 1), 0)]
    elif pursuer_action == SW:
        pursuer_pos = [max(min(pursuer_x + 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y - 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == WEST:
        pursuer_pos = [max(min(pursuer_x + dx,     WORLD_X - 1), 0), max(min(pursuer_y - 1 + dy, WORLD_Y - 1), 0)]
    elif pursuer_action == NW:
        pursuer_pos = [max(min(pursuer_x - 1 + dx, WORLD_X - 1), 0), max(min(pursuer_y - 1 + dy, WORLD_Y - 1), 0)]

    # evader action

    if np.random.binomial(1, gremlin) == 1:
        evader_action = np.random.choice(ACTIONS)

    if evader_action == NORTH:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y + dy, WORLD_Y - 1), 0)]
    elif evader_action == NE:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == EAST:
        evader_pos = [max(min(evader_x + dx, WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == SE:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y + 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == SOUTH:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y + dy, WORLD_Y - 1), 0)]
    elif evader_action == SW:
        evader_pos = [max(min(evader_x + 1 + dx, WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == WEST:
        evader_pos = [max(min(evader_x + dx, WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]
    elif evader_action == NW:
        evader_pos = [max(min(evader_x - 1 + dx, WORLD_X - 1), 0), max(min(evader_y - 1 + dy, WORLD_Y - 1), 0)]

    state = [pursuer_pos, evader_pos]
    return state


# play for an episode, return amount of time spent in the episode
# game keeps going until (1) evader reaches one of the escape squares or (2) pursuer catches evader
def episode(pursuer_q_value, evader_q_value, eps, gremlin, alpha):

    # initialize the counter that will track the total time taken for this episode
    t = 0

    # initialize state
    state = INIT_STATE
    pursuer_pos = PURSUER_START
    evader_pos = EVADER_START

    # choose initial pursuer and evader actions randomly
    pursuer_action = np.random.choice(ACTIONS)
    evader_action = np.random.choice(ACTIONS)

    pursuer_wins = False
    evader_wins = False
    draw = False
    result = ""

    while (not pursuer_wins) and (not evader_wins) and (not draw):

        # determine the next state
        next_state = step(state, pursuer_action, evader_action, gremlin)

        # choose the next pursuer action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:

            # eps% of the time, select action randomly
            next_pursuer_action = np.random.choice(ACTIONS)

        else:

            # (1-eps)% of the time, we select an action greedily
            # Select the action associated with the maximum q_value found among all q_values stored in values_
            # for each action_ and value_ stored in values_
            #   if the variable value_ is maximum in the list values_
            #     select the action_ associated with that q_value.
            pursuer_values_ = pursuer_q_value[pursuer_pos[0], pursuer_pos[1], :]
            next_pursuer_action = np.random.choice(
                [action_ for action_, value_ in enumerate(pursuer_values_) if (value_ == np.max(pursuer_values_)).any()])

        # choose the next evader action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:
            next_evader_action = np.random.choice(ACTIONS)
        else:
            evader_values_ = pursuer_q_value[evader_pos[0], evader_pos[1], :]
            next_evader_action = np.random.choice(
                [action_ for action_, value_ in enumerate(evader_values_) if (value_ == np.max(evader_values_)).any()])

        # set up reward structure
        if pursuer_pos == evader_pos:
            pursuer_reward = 1000
        else:
            pursuer_reward = -1
        if (evader_pos == ESCAPE_SQ1) or (evader_pos == ESCAPE_SQ2):
            evader_reward = 1000
        else:
            evader_reward = -1

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

        # update variables
        state = next_state
        pursuer_action = next_pursuer_action
        evader_action = next_evader_action
        pursuer_pos, evader_pos = state
        t += 1

        # determine game state
        pursuer_wins = (pursuer_pos == evader_pos)
        if pursuer_wins:
            result = PURSUER_WIN
        evader_wins = (evader_pos == ESCAPE_SQ1) or (evader_pos == ESCAPE_SQ2)
        if evader_wins:
            result = EVADER_WIN
        draw = (t == MAX_TIME)
        if draw:
            result = DRAW

    return t, state, result


def runner(eps, gremlin, alpha):

    pursuer_win_cnt = 0
    evader_win_cnt = 0
    draw_cnt = 0

    pursuer_q_value = np.zeros((WORLD_X, WORLD_Y, ACTION_SPACE_SIZE))
    evader_q_value = np.zeros((WORLD_X, WORLD_Y, ACTION_SPACE_SIZE))

    t0 = time.process_time()
    episode_limit = 1000
    ep = 0
    while ep < episode_limit:
        ep += 1
        t, s, r = episode(pursuer_q_value, evader_q_value, eps, gremlin, alpha)
        if r == PURSUER_WIN:
            pursuer_win_cnt = pursuer_win_cnt + 1
        elif r == EVADER_WIN:
            evader_win_cnt = evader_win_cnt + 1
        else:
            draw_cnt = draw_cnt + 1
    t1 = time.process_time()

    print("epsilon-greedy parameter = ", eps, ", noise parameter = ", gremlin, ", alpha = ", alpha)
    print("pursuer wins: ", pursuer_win_cnt, ", evader wins: ", evader_win_cnt, ", draws: ", draw_cnt)
    print("Time elapsed: ", t1-t0, " seconds")


if __name__ == '__main__':

    print("BASELINE SCENARIO")
    runner(0.2, 0.1, 0.5)
    print()
