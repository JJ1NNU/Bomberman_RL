import os
import pickle
import random
import time
import math

import numpy as np

from settings import *
from main import *
from . import auxiliary_functions as aux

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

cols = (COLS - 2)
cells = pow((COLS - 2), 2)
n = COLS - 1
assert COLS == ROWS

RANDOM_ACTION = .15

custom_bomb_state = []
BOMB_EVADE_STATES = 16

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    #
    #   For Retraining Existing Model
    #
    if self.train and os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)

    elif self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        # Setting up the model using the dimension reduction

        # First, compute the total number of disjoint states (after applying dimension reduction)

        number_of_agent_states = int(3 * n ** 2 / 16)  # for the agent, only a part of the tiles are valid positions
        number_of_coin_states = 1
        for k in range(1, n):
            if k % 2 == 0:
                number_of_coin_states += k // 2
            else:
                number_of_coin_states += k

        number_of_bomb_states = int((n - 1) ** 2 - (n / 2 - 1) ** 2) + 1  # this is the total number of valid positions
        # on the board plus one for the case of no bomb on the board
        number_of_crate_states = 2**4
        nr_states = number_of_agent_states * number_of_bomb_states * number_of_coin_states * number_of_crate_states
        shape = (nr_states, len(ACTIONS))
        self.logger.debug(f"New model has dimensions {shape}")
        self.Q = np.ones(shape=shape, dtype=np.float16)
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # update custom bomb state tracker
    update_custom_bomb_state(game_state)
    
    # todo Exploration vs exploitation
    if self.train and random.random() < RANDOM_ACTION:
        self.logger.debug("CB: Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("CB: Querying model for action.")
    
    agent_pos = game_state["self"][3]
    custom_coin_state = game_state['coins']
    if not len(custom_coin_state):
        if len(game_state['others']):
            other_positions = [game_state['others'][k][-1] for k in range(len(game_state['others']))]
            closest_agent = aux.index_of_closest_item(agent_position=agent_pos,
                                                      item_positions=other_positions)
            other_pos = game_state['others'][closest_agent][-1]
            custom_coin_state.append(other_pos)
        else:
            crate_pos = find_closest_crate(game_state)
            if crate_pos is not None:
                custom_coin_state.append(crate_pos)
    coin_index = aux.index_of_closest_item(agent_position=agent_pos, item_positions=custom_coin_state)
    bomb_positions = [bomb_attributes[0] for bomb_attributes in custom_bomb_state]
    bomb_index = aux.index_of_closest_item(agent_position=agent_pos, item_positions=bomb_positions)
    index, permutations = aux.state_to_index(game_state, custom_bomb_state, coin_index=coin_index, bomb_index=bomb_index,
                                             dim_reduce=True, include_bombs=True, include_crates=True)
    action_index = np.argmax(self.Q[index])
    permuted_action = ACTIONS[action_index]
    action_chosen = aux.revert_permutations(permuted_action, permutations)

    actions = self.Q[index]

    self.logger.debug(f"CB: Q[{index}]: {actions}")
    self.logger.debug(f"CB: Argmax-action of row is {permuted_action}")
    self.logger.debug(f"CB: Action chosen after reverting permutations {permutations}: {action_chosen}")

    return action_chosen

def update_custom_bomb_state(game_state):

    global custom_bomb_state

    # update timer
    updated_bombs = []
    for i in range(len(custom_bomb_state)):
        bomb = custom_bomb_state[i]
        new_bomb = ((bomb[0][0], bomb[0][1]), bomb[1] - 1)
        updated_bombs.append(new_bomb)

    custom_bomb_state.clear()
    for i in range(len(updated_bombs)):
        bomb = updated_bombs[i]
        custom_bomb_state.append(bomb)

    updated_bombs.clear()

    # add new to list
    new_bombs = []
    for i in range(len(game_state['bombs'])):
        bomb = game_state['bombs'][i]
        isIn = False
        for j in range(len(custom_bomb_state)):
            custom_bomb = custom_bomb_state[j]
            if bomb[0] == custom_bomb[0]:
                isIn = True
                break

        if not isIn:
            new_bombs.append(bomb)

    for i in range(len(new_bombs)):
        bomb = new_bombs[i]
        custom_bomb_state.append(bomb)

    new_bombs.clear()

    # remove old ones
    remove_bombs = []
    for i in range(len(custom_bomb_state)):
        bomb = custom_bomb_state[i]
        if bomb[1] <= -2:
            remove_bombs.append(bomb)

    for i in range(len(remove_bombs)):
        bomb = remove_bombs[i]
        # recheck this part
        custom_bomb_state.remove(bomb)

    remove_bombs.clear()

def find_closest_coin(game_state: dict):
    coins = game_state["coins"]
    agent = game_state["self"]
    
    if len(coins) <= 0:
        return None
        
    closest_coin = coins[0]
    closest_coin_dist = 1.797e+308 # max float value
    for i in range(len(coins)):
        coin_pos = coins[i]
        agent_pos = agent[3]
        euclid_dist = math.sqrt(pow((coin_pos[0] - agent_pos[0]), 2) + pow((coin_pos[1] - agent_pos[1]), 2))
        if euclid_dist < closest_coin_dist:
            closest_coin = coins[i]
            closest_coin_dist = euclid_dist
    
    return closest_coin

def find_closest_crate(game_state: dict):
    field = game_state["field"]
    agent = game_state["self"]
    field_shape = field.shape

    closest_crate = None
    closest_crate_dist = 1.797e+308 # max float value
    for x in range(field_shape[0]):
        for y in range(field_shape[1]):
            tile = field[x][y]
            if tile == 1: # 1 = crate
                agent_pos = agent[3]
                euclid_dist = math.sqrt(pow((x - agent_pos[0]), 2) + pow((y - agent_pos[1]), 2))
                if euclid_dist < closest_crate_dist:
                    closest_crate = (x, y)
                    closest_crate_dist = euclid_dist

    return closest_crate

def find_closest_bomb(game_state: dict):
    bombs = game_state["bombs"]
    agent = game_state["self"]

    if len(bombs) <= 0:
        return None

    closest_bomb = bombs[0]
    closest_bomb_dist = 1.797e+308 # max float value
    for i in range(len(bombs)):
        bomb_pos = bombs[i][0]
        agent_pos = agent[3]
        euclid_dist = math.sqrt(pow((bomb_pos[0] - agent_pos[0]), 2) + pow((bomb_pos[1] - agent_pos[1]), 2))
        if euclid_dist < closest_bomb_dist:
            closest_bomb = bombs[i]
            closest_bomb_dist = euclid_dist

    return closest_bomb

def is_bomb_under_players_feet(game_state):
    agent_pos = game_state["self"][3]
    bombs = game_state["bombs"]
    if len(bombs) == 0:
        return False
    for bomb in bombs:
        bomb_pos = bomb[0]
        if agent_pos[0] == bomb_pos[0] and agent_pos[1] == bomb_pos[1]:
            return True
    return False

def compute_evade_bomb_index(game_state):
    if is_bomb_under_players_feet(game_state):
        left = can_escape_left(game_state)
        right = can_escape_right(game_state)
        up = can_escape_up(game_state)
        down = can_escape_down(game_state)
        index = left * 8 + right * 4 + up * 2 + down * 1
        return index

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def can_escape_left(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]

    field = game_state["field"]

    for i in range(1, 4):

        # check for out of bounds
        if ag_x - i < 0 or ag_x - i >= ROWS:
            return 0

        # check infront
        tile = field[ag_x - i][ag_y]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0

        # check top
        tile = field[ag_x - i][ag_y - 1]
        if tile == 0:
            return 1

        # check bottom
        tile = field[ag_x - i][ag_y + 1]
        if tile == 0:
            return 1

    # check for out of bounds
    if ag_x - 4 < 0 or ag_x - 4 >= ROWS:
        return 0

    # check last tile that is right out of range of bomb
    tile = field[ag_x - 4][ag_y]
    if tile == 0:
        return 1

    return 0

def can_escape_right(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]

    field = game_state["field"]

    for i in range(1, 4):

        # check for out of bounds
        if ag_x + i < 0 or ag_x + i >= ROWS:
            return 0

        # check infront
        tile = field[ag_x + i][ag_y]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0

        # check top
        tile = field[ag_x + i][ag_y - 1]
        if tile == 0:
            return 1

        # check bottom
        tile = field[ag_x + i][ag_y + 1]
        if tile == 0:
            return 1

    # check for out of bounds
    if ag_x + 4 < 0 or ag_x + 4 >= ROWS:
        return 0

    # check last tile that is right out of range of bomb
    tile = field[ag_x + 4][ag_y]
    if tile == 0:
        return 1

    return 0

def can_escape_up(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]

    field = game_state["field"]

    for i in range(1, 4):

        # check for out of bounds
        if ag_y - i < 0 or ag_y - i >= COLS:
            return 0

        # check infront
        tile = field[ag_x][ag_y - i]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0

        # check left
        tile = field[ag_x - 1][ag_y - i]
        if tile == 0:
            return 1

        # check right
        tile = field[ag_x + 1][ag_y - i]
        if tile == 0:
            return 1

    # check for out of bounds
    if ag_y - 4 < 0 or ag_y - 4 >= ROWS:
        return 0

    # check last tile that is right out of range of bomb
    tile = field[ag_x][ag_y - 4]
    if tile == 0:
        return 1

    return 0

def can_escape_down(game_state):
    agent_pos = game_state["self"][3]
    ag_x = agent_pos[0]
    ag_y = agent_pos[1]

    field = game_state["field"]

    for i in range(1, 4):

        # check for out of bounds
        if ag_y + i < 0 or ag_y + i >= COLS:
            return 0

        # check infront
        tile = field[ag_x][ag_y + i]
        if tile != 0:   # tile is crate (1) or stone wall (-1)
            return 0

        # check left
        tile = field[ag_x - 1][ag_y + i]
        if tile == 0:
            return 1

        # check right
        tile = field[ag_x + 1][ag_y + i]
        if tile == 0:
            return 1

    # check for out of bounds
    if ag_y + 4 < 0 or ag_y + 4 >= ROWS:
        return 0

    # check last tile that is right out of range of bomb
    tile = field[ag_x][ag_y + 4]
    if tile == 0:
        return 1

    return 0
