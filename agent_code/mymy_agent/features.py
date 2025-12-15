import numpy as np
import scipy.signal as sc

tile_features = np.eye(5)

blast_masks = np.array([
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ],
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]
])

def get_features(game_state: dict) -> tuple[np.ndarray, ...]:
    """
    Returns the features from the game state.
    """
    step = np.array([game_state["step"]])
    closest_coin = get_closest_coin(game_state)
    closest_opponent = get_closest_opponent(game_state)
    nearby_opponents = count_nearby_opponents(np.array([opponent[3] for opponent in game_state["others"]]), np.array(game_state["self"][3]))
    tiles, blast_mask = get_tile_type(game_state["field"], game_state["self"][3])
    nearby_crates = count_nearby_crates(game_state["field"], game_state["self"][3], blast_mask)
    dense_crate_directions = get_highest_crate_density_direction(game_state["field"], game_state["self"][3])

    return step, closest_coin, nearby_crates, closest_opponent, nearby_opponents, tiles, dense_crate_directions


def get_closest_coin(game_state: dict) -> np.ndarray:
    """
    Relative coordinates to the closest coin, or (np.inf, np.inf) if no coin is present.
    """

    coins = np.array(game_state["coins"])
    if len(coins) == 0:
        return np.array([16, 16])
    user_coords = game_state["self"][3]
    relative_coins = coins - user_coords
    distances = np.abs(relative_coins).sum(axis=1)
    closest_coin = relative_coins[np.argmin(distances)]
    return closest_coin


def get_closest_opponent(game_state: dict) -> np.ndarray:
    """
    Relative coordinates to the closest opponent, or (np.inf, np.inf) if no opponent is present.
    """
    if len(game_state["others"]) == 0:
        return np.array([16, 16])

    opponent_coords = np.array([opponent[3] for opponent in game_state["others"]])

    user_coords = np.array(game_state["self"][3])
    relative_opponents = opponent_coords - user_coords
    distances = np.abs(relative_opponents).sum(axis=1)
    closest_opponent = relative_opponents[np.argmin(distances)]
    return closest_opponent


def count_nearby_opponents(opponent_coords: np.ndarray, user_coords: np.ndarray) -> np.ndarray:
    """
    Returns the amount of opponents in a 7x7 window around the user.
    """
    if len(opponent_coords) == 0:
        return np.array([0])
    relative_opponents = np.abs(opponent_coords - user_coords)
    nearby_opponents = (relative_opponents <= 3).all(axis=1).sum()

    return np.array([nearby_opponents])


def get_tile_type(field: np.ndarray, user_coords: np.ndarray) -> np.ndarray:
    """
    Returns a vector with the tile type and the blast mask.
    """
    window = field[user_coords[0]-1:user_coords[0]+2, user_coords[1]-1:user_coords[1]+2] == -1
    n_walls = window.sum()
    if n_walls == 6:
        return tile_features[0], blast_masks[0]
    elif n_walls == 5:
        return tile_features[1], blast_masks[1]
    elif n_walls == 4:
        if window[-1,-1]:
            return tile_features[2], blast_masks[2]
        else:
            return tile_features[3], blast_masks[3]
    elif n_walls == 2:            
        return tile_features[4], (blast_masks[3] if window[0,1] else blast_masks[4])
    

def get_highest_crate_density_direction(field: np.ndarray, user_coords: np.ndarray) -> np.ndarray:
    """
    Convolves the field with a 3x3 summing window and returns the direction to the maximum of the convolved field.
    """
    window = np.ones((3,3))
    convolved = sc.convolve2d((field == 1), window, mode="same")
    max_index = np.argmax(convolved)
    distance = np.array([max_index // field.shape[1] - user_coords[0], max_index % field.shape[1] - user_coords[1]])
    return np.sign(distance)


def count_nearby_crates(field: np.ndarray, user_coords: np.ndarray, blast_mask: np.ndarray) -> np.ndarray:
    """
    Returns the amount of crates a bomb would hit in that position.
    """
    window = field[np.r_[user_coords[0]-3:user_coords[0]+4]][:, np.r_[user_coords[1]-3:user_coords[1]+4]]
    crates = ((window == 1) * blast_mask).sum()
    return np.array([crates])


def get_nearby_bombs(game_state: dict) -> np.ndarray:
    """
    Returns the relative coordinates of the bombs in a 7x7 window around the user.
    """
    user_coords = np.array(game_state["self"][3])
    bomb_coords = np.array([bomb[0] for bomb in game_state["bombs"]])
    if len(bomb_coords) == 0:
        return 16 * np.ones((4,2))
    
    relative_bombs = bomb_coords - user_coords

    bombs = 16 * np.ones((4,2))
    bombs[:len(relative_bombs)] = relative_bombs

    return bombs