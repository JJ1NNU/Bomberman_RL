import pickle
import numpy as np
from . import features as base_features
from typing import Callable

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


values = {"free": 0, "wall": 1, "crate": 2, "coin": 4, "player": 8,
           "opponent": 16, "has_bomb": 32, "bomb0": 64, "bomb1": 128,
             "bomb2": 256, "bomb3": 512, "explosion": 1024}

binary_values = {"wall": 0, "crate": 1, "coin": 2, "player": 3,
           "opponent": 4, "has_bomb": 5, "bomb0": 6, "bomb1": 7,
             "bomb2": 8, "bomb3": 9, "explosion": 10}

WIN_RADIUS = 3
WIN_SIZE = 2 * WIN_RADIUS + 1

N_NEIGHBOURS_RATIO = 0.01
MAX_N_NEIGHBOURS = 10000


def setup(self, intialization_function: Callable, resume_training: bool = True):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param intialization_function: Function that initializes the self.features and self.Q_table
    :param resume_training: If True, the Q-table and features are loaded from the last training session. If False, the Q-table and features are initialized from scratch.
    """

    # In a 7x7 window, max 4 bombs, 3 opponents, 40 coins, 40 crates, 40 explosions

    if not self.train or (self.train and resume_training) or (self.train and self.eval_mode):
        with open("Q_table.pkl", "rb") as f:
            self.Q_table = pickle.load(f)
            print(f"Loaded Q table with {len(self.Q_table)} states")
        with open("features.pkl", "rb") as f:
            self.features = pickle.load(f)

    else:
        intialization_function(self)
        # Neighbours are not used anymore, keeping for reference
        #self.neighbours = np.zeros((1, 0), dtype=np.int16)  # (N, 50)
        #self.neighbour_distances = np.zeros((1, 0))         # (N, 50)

    # Used in places where a calculation has to be done for all but the first axis
    self.feature_axes = tuple(tuple([i for i in range(1,self.features[j].ndim)]) for j in range(len(self.features)))


def select_move(q_values: np.ndarray, allowed_moves: np.ndarray) -> str:
    """
    Selects best move that is not invalid.
    """

    best_moves = np.argsort(q_values)[::-1]

    for move in best_moves:
        if allowed_moves[move]:
            return ACTIONS[move]
        
    print("No safe move found") # Death is imminent
    return ACTIONS[best_moves[0]]



def get_valid_moves(field: np.ndarray, user_coords: tuple[int, int], bomb_possible: bool = True, allow_wait: bool = True) -> np.ndarray:
    """
    Returns list of valid movement moves
    bomb_possible: Can be taken from the game state but can also be set to False if we don't want to allow the agent to place a bomb.
    """
    
    left = field[user_coords[0] - 1, user_coords[1]]
    right = field[user_coords[0] + 1, user_coords[1]]
    up = field[user_coords[0], user_coords[1] - 1]
    down = field[user_coords[0], user_coords[1] + 1]

    move_tiles = np.array([up, right, down, left])
    valid_moves = move_tiles == 0

    return np.append(valid_moves, [allow_wait, bomb_possible])


def get_safe_moves(game_state: dict, moves: np.ndarray[bool]) -> np.ndarray[bool]:
    """
    Returns list of safe moves where the agent does not die in the coming 4 steps.
    Implements a recursive function to check for safe moves.
    A bomb explodes on the 4th step, the explosion vanishes on the 6th step.
    """
    spread_directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    def is_safe(
            field: np.ndarray, 
            bombs: list[tuple[tuple[int, int], int]], 
            explosion_map: np.ndarray, 
            user_coords: tuple[int, int],  
            move: str, 
            step: int = 0
        ) -> bool:
        """
        Recursive function to check if a move is safe.
        """
        if step == maxBombTimer + 1:
            return True

        if move == "UP":
            next_coords = (user_coords[0], user_coords[1] - 1)
        elif move == "RIGHT":
            next_coords = (user_coords[0] + 1, user_coords[1])
        elif move == "DOWN":
            next_coords = (user_coords[0], user_coords[1] + 1)
        elif move == "LEFT":
            next_coords = (user_coords[0] - 1, user_coords[1])
        else:
            next_coords = user_coords

        # Assume that the move is valid
        new_explosion_map = np.maximum(explosion_map - 1, 0)
        exploding_bombs = [bomb for bomb in bombs if bomb[1] == 0]
        new_field = field.copy()
        for bomb in exploding_bombs:
            new_explosion_map[bomb[0]] = 2
            new_field[bomb[0]] = 0
            for direction in spread_directions:
                for r in range(1, 4): # 3 length explosion in all directions
                    new_coords = bomb[0] + r * direction
                    if new_field[*new_coords] == -1: # Hit a wall, validity of coordinate doesn't have to be checked as there is a wall around the field
                        break
                    new_explosion_map[*new_coords] = 2
                    new_field[*new_coords] = 0
        
        if new_explosion_map[*next_coords] != 0:
            return False
        
        new_bombs = [(bomb[0], bomb[1] - 1) for bomb in bombs if bomb[1] > 0]
        if move == "BOMB":
            new_bombs.append((user_coords, 3))
        new_field = add_coords_to_matrix(new_field, np.array([bomb[0] for bomb in new_bombs]), value=1)

        new_moves = get_valid_moves(new_field, next_coords, allow_wait=True, bomb_possible=False)
        safe = False
        for move_id, valid in enumerate(new_moves):
            if not valid:
                continue
            if is_safe(new_field, new_bombs, new_explosion_map, next_coords, ACTIONS[move_id], step + 1):
                safe = True
                break
        return safe
        

    field = game_state['field']
    bombs = game_state['bombs']
    if len(bombs) == 0:
        if np.sum(game_state['explosion_map']) == 0 and not moves[5]:
            return moves
        maxBombTimer = 2
    else:
        maxBombTimer = np.max([bomb[1] for bomb in bombs])
    explosion_map = game_state['explosion_map'].copy()
    explosion_map[explosion_map == 1] = 2 # The explosion map provided in the game state is stupid, 
                                          # it gives 1 when there is still another step where the explosion is dangerous
                                          # And 0 where the explosion is dangerous in this step as well as not dangerous
                                          # We set the 1s to 2 so we can calculate as before, but the places where
                                          # 1s are set to 0 are not important because we are alive and in the next step it would 
                                          # go to 0 anyway.
    user_coords = game_state['self'][3]

    for move_id, valid in enumerate(moves):
        if not valid:
            continue
        if ACTIONS[move_id] == "BOMB":
            maxBombTimer = 4 # This bomb will be placed at the first step, so to make sure it is taken into account, the maxBombTimer is set to 4
        moves[move_id] = is_safe(field, bombs, explosion_map, user_coords, ACTIONS[move_id])
    return moves


def coords_to_matrix(coords, size, value=1) -> np.ndarray:
    matrix = np.zeros((size, size))
    matrix[coords[:,0], coords[:,1]] = value
    return matrix


def add_coords_to_matrix(matrix, coords, value=1) -> np.ndarray:
    if len(coords) == 0: return matrix
    matrix[coords[...,0], coords[...,1]] += value
    return matrix


def state_to_matrix(game_state: dict) -> np.ndarray:
    """
    Binary values for each object on the field
    0: free space
    1: wall
    2: crate
    4: coin
    8: player agent
    16: opponent agent
    32: agent has bomb

    64: bomb at 0
    128: bomb at 1
    256: bomb at 2
    512: bomb at 3

    1024: explosion
    """


    user_coords = np.array(game_state['self'][3]) - 1
    coin_coords = np.array(game_state['coins']) - 1
    bomb_info = game_state['bombs']
    player_info = game_state['others']

    bomb_coords = np.array([bomb[0] for bomb in bomb_info]) - 1
    bomb_timers = np.array([bomb[1] for bomb in bomb_info])
    player_coords = np.array([player[3] for player in player_info]) - 1
    has_bomb = np.array([player[2] for player in player_info]) - 1
    
    field = game_state['field'][1:-1, 1:-1]
    explosion_map = np.uint8(game_state['explosion_map'])[1:-1,1:-1]


    state_matrix = values["wall"] * (field == -1) + values["crate"] * (field == 1) + values["explosion"] * explosion_map

    state_matrix = add_coords_to_matrix(state_matrix, coin_coords, value = values["coin"])
    state_matrix = add_coords_to_matrix(state_matrix, player_coords, value = values["opponent"] + has_bomb * values["has_bomb"])
    state_matrix = add_coords_to_matrix(state_matrix, bomb_coords, value = 2 ** (bomb_timers) * values["bomb0"])

    state_matrix[user_coords[0], user_coords[1]] += values["player"] + values["has_bomb"] * game_state['self'][2]

    return np.int16(state_matrix) # Must be unsigned for unpacking


def get_standardized_state(state_matrix: np.ndarray) -> np.ndarray:
    """
    Function to bring the state matrix to a standard form.
    This allows the agent to be rotation and flip invariant.
    """
    # NOTE: this is not the fastest way of doing it, but i'm lazy :D
    # three flips is equivalent to rotation + flip

    bits = unpackbits(state_matrix)
    user_coords = np.argwhere(unpackbits(state_matrix)[binary_values["player"]] == 1)[0]
    size = state_matrix.shape[0]

    if user_coords[0] > size//2:
        state_matrix = np.flip(state_matrix, axis=0)
        user_coords[0] = size - user_coords[0] - 1 

    if user_coords[1] > size//2:
        state_matrix = np.flip(state_matrix, axis=1)
        user_coords[1] = size - user_coords[1] - 1

    if user_coords[0] > user_coords[1]:
        state_matrix = state_matrix.T
        user_coords = user_coords[::-1]
    
    return state_matrix, user_coords


def get_standarized_state_dict(game_state: dict) -> dict:
    """
    Returns a standarized state matrix and the user coordinates.
    """
    game_state = game_state.copy()
    field = game_state['field'].copy()
    explosion_map = game_state['explosion_map'].copy()
    user_coords = game_state['self'][3]
    n_coins = len(game_state["coins"])
    n_others = len(game_state["others"])

    coins = game_state['coins']
    coord_list = np.array(
                [user_coords]
                + game_state["coins"] 
                + [other[3] for other in game_state['others']] 
                + [bomb[0] for bomb in game_state['bombs']])
    

    
    size = field.shape[0]
    transformations = [False, False, False]

    if coord_list[0, 0] > size//2: # Horizontal flip (in the board, not the matrix)
        field = np.flip(field, axis=0)
        explosion_map = np.flip(explosion_map, axis=0)
        coord_list[:, 0] = size - coord_list[:, 0] - 1 
        transformations[0] = True

    if coord_list[0, 1] > size//2: # Vertical flip
        field = np.flip(field, axis=1)
        explosion_map = np.flip(explosion_map, axis=1)
        coord_list[:, 1] = size - coord_list[:, 1] - 1
        transformations[1] = True

    if coord_list[0, 0] > coord_list[0, 1]:
        field = field.T
        explosion_map = explosion_map.T
        coord_list = coord_list[:, ::-1]
        transformations[2] = True

    if True in transformations:
        game_state["field"] = field
        game_state["explosion_map"] = explosion_map
        game_state["self"] = game_state["self"][:3] + (tuple(coord_list[0]),)

        coins = [tuple(coord_list[i]) for i in range(1,n_coins+1)]
        game_state["coins"] = coins
        others = [others[:3] + (tuple(coord_list[1+n_coins+i]),) for i, others in enumerate(game_state["others"])]
        game_state["others"] = others
        bombs = [(tuple(coord_list[1+n_coins+n_others+i]), bomb[1]) for i, bomb in enumerate(game_state["bombs"])]
        game_state["bombs"] = bombs

    return game_state, transformations


def forward_move_transformations(action: str, transformations: np.ndarray) -> str:
    if action == "BOMB" or action == "WAIT":
        return action
    
    if transformations[0]:
        if action == "LEFT":
            action = "RIGHT"
        elif action == "RIGHT":
            action = "LEFT"
    
    if transformations[1]:
        if action == "UP":
            action = "DOWN"
        elif action == "DOWN":
            action = "UP"
    
    if transformations[2]:
        if action == "UP":
            action = "LEFT"
        elif action == "LEFT":
            action = "UP"
        elif action == "RIGHT":
            action = "DOWN"
        elif action == "DOWN":
            action = "RIGHT"

    return action


def revert_move_transformations(action: str, transformations: np.ndarray) -> str:
    if action == "BOMB" or action == "WAIT":
        return action
    
    if transformations[2]:
        if action == "UP":
            action = "LEFT"
        elif action == "LEFT":
            action = "UP"
        elif action == "RIGHT":
            action = "DOWN"
        elif action == "DOWN":
            action = "RIGHT"
    
    if transformations[0]:
        if action == "LEFT":
            action = "RIGHT"
        elif action == "RIGHT":
            action = "LEFT"
    
    if transformations[1]:
        if action == "UP":
            action = "DOWN"
        elif action == "DOWN":
            action = "UP"

    return action


def unpackbits(x: np.ndarray, num_bits: int = 16) -> np.ndarray:
    """
    Unpacks an array of integers into a binary matrix. 
    Stolen from stackoverflow as numpy's unpackbits only works for uint8
    """
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits]).transpose([2,0,1])


def remove_walls(state_matrix: np.ndarray) -> np.ndarray:
    flat_state = state_matrix.flatten()
    return flat_state[flat_state != 1]


def add_walls(flat_state: np.ndarray) -> np.ndarray:
    assert flat_state.dtype == np.int16, "Flat state must be a int16 array"
    # Length of flat state if n even: 3 / 4 n^2
    trialEven = np.sqrt(len(flat_state) * 4 / 3)
    # Length of flat state if n odd: 3 / 4 (n - 1)^2 + 2(n - 1) + 1
    trialOdd = (-2 + np.sqrt(4 - 3 * (1 - len(flat_state)))) / (3 / 2) + 1 # Quadratic formula for (n - 1)
    if trialOdd == int(trialOdd):
        size = int(trialOdd)
    elif trialEven == int(trialEven):
        size = int(trialEven)
    else:
        raise ValueError("Flat state length does not correspond to a square matrix")
    state_matrix = np.ones((size, size), dtype=np.int16)
    size_with_walls = size + (size + 1) // 2
    for i in range(size):
        if i % 2 == 0:
            state_matrix[i] = flat_state[size_with_walls*i//2 : size_with_walls*i//2 + size]
        else:
            state_matrix[i,::2] = flat_state[size_with_walls*(i-1)//2 + size: size_with_walls*(i-1)//2 + size_with_walls]

    return state_matrix


def get_player_window(state_matrix: np.ndarray, user_coords: np.ndarray, window_radius: int = 7) -> np.ndarray:
    """
    Function to get a square window around the player.
    """
    state_matrix = np.pad(state_matrix, window_radius, constant_values=values["wall"])	
    window = state_matrix[user_coords[0] : user_coords[0] + 2 * window_radius + 1,
                             user_coords[1] : user_coords[1] + 2 * window_radius + 1]
    
    return window

    

def Q_regression(self, features: np.ndarray, is_known: bool) -> np.ndarray:
    """
    Function to estimate the Q-value for a feature. Based on similar features in the Q-table.
    Finds K nearest neighbors (eulerian distance) in the states matrix and returns the averaged Q-value for each action.
    """
    if len(self.Q_table) == 0:
        return np.zeros(6)
    elif len(self.Q_table) == 1:
        return self.Q_table[0]

    begin = 1 if is_known else 0

    distances = base_features.get_distances(self, features)

    n_neighbours = min(int(N_NEIGHBOURS_RATIO * len(self.Q_table)), MAX_N_NEIGHBOURS)

    neighbour_idx = np.argsort(distances)[begin : begin + n_neighbours]
    distances = distances[neighbour_idx]

    weights = 1 / distances
    weights /= weights.sum()
    action_values = self.Q_table[neighbour_idx].T @ weights

    return action_values


def store_state(
        self, 
        old_state: dict, 
        new_state: dict, 
        get_features: Callable[[dict], tuple[np.ndarray, ...]]
    ) -> tuple[np.ndarray, np.ndarray, int, int]:

    old_state["step"] -= 1
    old_features = self.new_features if old_state["step"] != 0 else get_features(old_state)
    new_features = get_features(new_state)
    

    old_idx, old_is_known = base_features.get_state_id(self, old_features)
    
    if old_state["step"] == 0 and not old_is_known:
        #update_neighbours(self, old_features)
        for i in range(len(self.features)):
            self.features[i] = np.concatenate((self.features[i], old_features[i][np.newaxis]), axis=0)
                
        self.Q_table = np.concatenate((self.Q_table, np.zeros((1, 6))), axis=0)
    
    old_idx, old_is_known = base_features.get_state_id(self, old_features)
    new_idx, new_is_known = base_features.get_state_id(self, new_features)

    if not new_is_known:
        #update_neighbours(self, new_features)
        for i in range(len(self.features)):
            self.features[i] = np.concatenate((self.features[i], new_features[i][np.newaxis]), axis=0)
        self.Q_table = np.concatenate((self.Q_table, np.zeros((1, 6))), axis=0)

    return old_features, new_features, old_idx, new_idx


def epsion_greedy(self, initial_eps = 0.9, final_eps = 0.1, decay = 0.01) -> float:
    """
    Returns the value of epsilon for the current step.
    Exponential decay from initial_eps to final_eps.
    """
    return final_eps + (initial_eps - final_eps) * np.exp(- decay * self.rounds)


def get_safe_moves_from_state(game_state: dict) -> np.ndarray:

    field = game_state['field'].copy()
    field = add_coords_to_matrix(field, np.array([other[3] for other in game_state['others']]), value=1)
    field = add_coords_to_matrix(field, np.array([bomb[0] for bomb in game_state['bombs']]), value=1)

    valid_moves = get_valid_moves(
        field, 
        game_state['self'][3], 
        bomb_possible=game_state['self'][2], 
        allow_wait=True
    )

    safe_moves = get_safe_moves(game_state, valid_moves)

    return safe_moves


