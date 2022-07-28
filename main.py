import numpy as np

# dimensions
WORLD_HEIGHT = 2
WORLD_WIDTH = 2
MAX_TIME = 5000

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
PREDATOR_START = [0, 0]
PREY_START = [WORLD_HEIGHT - 1, WORLD_WIDTH - 1]
ESCAPE_SQ1 = [WORLD_HEIGHT - 1, 0]
ESCAPE_SQ2 = [0, WORLD_WIDTH - 1]
INIT_STATE = [PREDATOR_START, PREY_START]


# This function defines how the agents move on the grid.
# gremlin represents the probability that strategically selected action is replaced with a random action
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, pred_action, prey_action, gremlin):
    pred_pos, prey_pos = state
    pred_x = pred_pos[0]
    pred_y = pred_pos[1]
    prey_x = prey_pos[0]
    prey_y = prey_pos[1]
    dx = 0
    dy = 0

    # predator action

    if np.random.binomial(1, gremlin) == 1:
        pred_action = np.random.choice(ACTIONS)

    if pred_action == NORTH:
        pred_pos = [max(min(pred_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y + dy,     WORLD_WIDTH - 1), 0)]
    elif pred_action == NE:
        pred_pos = [max(min(pred_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif pred_action == EAST:
        pred_pos = [max(min(pred_x + dx,     WORLD_HEIGHT - 1), 0), max(min(pred_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif pred_action == SE:
        pred_pos = [max(min(pred_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif pred_action == SOUTH:
        pred_pos = [max(min(pred_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y + dy,     WORLD_WIDTH - 1), 0)]
    elif pred_action == SW:
        pred_pos = [max(min(pred_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif pred_action == WEST:
        pred_pos = [max(min(pred_x + dx,     WORLD_HEIGHT - 1), 0), max(min(pred_y - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif pred_action == NW:
        pred_pos = [max(min(pred_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(pred_y - 1 + dy, WORLD_WIDTH - 1), 0)]

    # prey action

    if np.random.binomial(1, gremlin) == 1:
        prey_action = np.random.choice(ACTIONS)

    if prey_action == NORTH:
        prey_pos = [max(min(prey_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y + dy,     WORLD_WIDTH - 1), 0)]
    elif prey_action == NE:
        prey_pos = [max(min(prey_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif prey_action == EAST:
        prey_pos = [max(min(prey_x + dx,     WORLD_HEIGHT - 1), 0), max(min(prey_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif prey_action == SE:
        prey_pos = [max(min(prey_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif prey_action == SOUTH:
        prey_pos = [max(min(prey_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y + dy,     WORLD_WIDTH - 1), 0)]
    elif prey_action == SW:
        prey_pos = [max(min(prey_x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif prey_action == WEST:
        prey_pos = [max(min(prey_x + dx,     WORLD_HEIGHT - 1), 0), max(min(prey_y - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif prey_action == NW:
        prey_pos = [max(min(prey_x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(prey_y - 1 + dy, WORLD_WIDTH - 1), 0)]

    state = [pred_pos, prey_pos]

    return state


# play for an episode, return amount of time spent in the episode
def episode(pred_q_value, prey_q_value, eps, gremlin, alpha):

    # initialize the counter that will track the total time steps in this episode
    time = 0

    # initialize grid
    # fig, ax = plt.subplots(1, 1, figsize=(WORLD_HEIGHT, WORLD_WIDTH))
    # plt.meshgrid(np.arange(0, WORLD_HEIGHT, 1), np.arange(0, WORLD_WIDTH, 1))

    # initialize state
    state = INIT_STATE
    pred_pos = PREDATOR_START
    prey_pos = PREY_START

    # choose an initial action based on epsilon-greedy algorithm (or when first starting a simulation)
    if np.random.binomial(1, eps) == 1 or time == 0:

        # eps% of the time, select actions randomly
        # TODO - Randomness of prey and predator actions should be made independent
        pred_action = np.random.choice(ACTIONS)
        prey_action = np.random.choice(ACTIONS)

    else:

        # (1-eps)% of the time, we select an action greedily
        # Select the action associated with the maximum q_value found among all q_values stored in values_
        # Algorithmically:
        # for each action_ and value_ stored in values_
        #   if the variable value_ is maximum in the list values_
        #     select the action_ associated with that q_value.

        pred_values_ = pred_q_value[pred_pos[0], pred_pos[1], :]
        pred_action = np.random.choice(
            [action_ for action_, value_ in enumerate(pred_values_) if (value_ - np.max(pred_values_)).all()])

        prey_values_ = prey_q_value[prey_pos[0], prey_pos[1], :]
        prey_action = np.random.choice(
            [action_ for action_, value_ in enumerate(prey_values_) if (value_ - np.max(prey_values_)).all()])

    # game keeps going until (1) prey reaches one of the escape squares or (2) predator catches prey
    # added a max_time to prevent an infinite loop; we'll consider this a successful evasion
    while prey_pos != ESCAPE_SQ1 and prey_pos != ESCAPE_SQ2 and pred_pos != prey_pos and time < MAX_TIME:

        # determine the next state
        next_state = step(state, pred_pos, prey_pos, gremlin)

        # choose the next action based on epsilon-greedy algorithm
        # TODO - Randomness of prey and predator actions should be made independent
        if np.random.binomial(1, eps) == 1:

            next_pred_action = np.random.choice(ACTIONS)
            next_prey_action = np.random.choice(ACTIONS)

        else:

            pred_values_ = pred_q_value[pred_pos[0], pred_pos[1], :]
            next_pred_action = np.random.choice(
                [action_ for action_, value_ in enumerate(pred_values_) if (value_ == np.max(pred_values_)).any()])

            prey_values_ = pred_q_value[prey_pos[0], prey_pos[1], :]
            next_prey_action = np.random.choice(
                [action_ for action_, value_ in enumerate(prey_values_) if (value_ == np.max(prey_values_)).any()])

        # SARSA update - For more info about SARSA algorithm, please refer to p. 129 of the S&B textbook.
        reward = -1
        pred_q_value[pred_pos[0], pred_pos[1], pred_action] = \
            pred_q_value[pred_pos[0], pred_pos[1], next_pred_action] \
            + alpha * (reward + pred_q_value[pred_pos[0], pred_pos[1], next_pred_action]
                       - pred_q_value[pred_pos[0], pred_pos[1], next_pred_action])
        prey_q_value[prey_pos[0], prey_pos[1], prey_action] = \
            prey_q_value[prey_pos[0], prey_pos[1], next_prey_action] \
            + alpha * (reward + prey_q_value[prey_pos[0], prey_pos[1], next_prey_action]
                       - prey_q_value[prey_pos[0], prey_pos[1], next_prey_action])
        state = next_state
        pred_action = next_pred_action
        prey_action = next_prey_action
        time += 1

    print(time, state)

    return time, state


# plots the figure average time step vs number of episodes
def runner(eps, gremlin, alpha):
    episode_limit = 1

    pred_q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, ACTION_SPACE_SIZE))
    prey_q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, ACTION_SPACE_SIZE))
    steps = []
    ep = 0

    while ep < episode_limit:
        steps.append(episode(pred_q_value, prey_q_value, eps, gremlin, alpha))
        ep += 1


if __name__ == '__main__':
    runner(0.2, 0.1, 0.5)
