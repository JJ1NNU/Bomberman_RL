import numpy as np
from typing import List
import settings as s

def convert_pos(position_tuple: tuple, n, is_a_coin=False):
    """
    I don't even know anymore
    """

    x, y = position_tuple
    if x == y == 0:  # This case shouldn't occur in the real game, because (0,0) is a wall. However, we can use this,
        return 0     # to mark the absence of any bomb or coin. It will be treated as a bomb/coin being placed at (0,0) and should have index 0.
    conv_pos = 0
    subtractor = 0
    conv_pos += ((y-1)//2) * (n//2) + ((y-1) - (y-1)//2) * (n-1)
    if y % 2 == 0:
        conv_pos += (x - x//2)
    else:
        conv_pos += x

    if is_a_coin:
        for y_k in range(y+1):
            subtractor += y_k - 1
            if y_k % 2 == 0:
                subtractor -= (y_k-1)//2

    conv_pos -= subtractor
    return int(conv_pos)


def index_of_closest_item(agent_position: tuple, item_positions: List[tuple]):
    d_min = s.ROWS+s.COLS  # this is larger than every possible distance between agent and coin
    i_min = -1
    if not len(item_positions):
        return None
    for i in range(len(item_positions)):
        d = distance_from_item(agent_position, item_positions[i])
        if d < d_min:
            i_min = i
            d_min = d
    if i_min < 0:
        return None
        #raise ValueError("No coin can be found close to the agent.")
    return i_min


def state_to_index(game_state: dict, custom_bomb_state, coin_index=None, bomb_index=None, dim_reduce=False, include_bombs=False, include_crates=False):
    """
    This function is a bit messy and even if it would work, it would only be applicable for this very simple setup and
    can get very complicated. We should definitely look for a better feature extraction for more complex states.
    """
    arena_height, arena_width = np.shape(game_state["field"])
    assert arena_height == arena_width
    n = arena_height - 1
    assert n % 2 == 0

    agent_position = game_state["self"][-1]
    if len(game_state["coins"]) != 0:
        if coin_index is None:  # if there is no coin_index given...
            # ...find the closest coin
            coin_index = index_of_closest_item(agent_position, game_state["coins"])
        coin_position = game_state["coins"][coin_index]  # use the coins position
    else:
        coin_position = (0, 0)  # there are no coins and the game should end (in single player round)
        # remark: similar to the bomb situation, the position (0,0) of a coin can be regarded as the case, where there
        # is no coin left on the board. In this case, the game could still go on (other agents might still be alive),
        # but the training should ignore the coin. The state needs to be numbered though, for which we need this extra
        # position, corresponding to the digit 0 in the index of the state.

    if include_bombs:
        if len(custom_bomb_state):
            if bomb_index is None:
                bomb_index = index_of_closest_item(agent_position,
                                                   [custom_bomb_state[k][0] for k in range(len(custom_bomb_state))])

            bomb_position = custom_bomb_state[bomb_index][0]
        else:
            bomb_position = (0, 0)  # same as above. No bomb -> bomb_position = (0,0) corresponds to index 0.

    permutations = []
    if dim_reduce:  # reduce the dimension of the Q matrix by exploiting mirror symmetry
        if agent_position[0] > n//2:
            agent_position_0 = n//2 - (agent_position[0] - (n//2))
            agent_position = (agent_position_0, agent_position[1])
            if coin_position != (0, 0):
                coin_position_0 = n//2 - (coin_position[0] - (n//2))
                coin_position = (coin_position_0, coin_position[1])
            if include_bombs and bomb_position != (0, 0):
                bomb_position_0 = n//2 - (bomb_position[0] - (n//2))
                bomb_position = (bomb_position_0, bomb_position[1])
            permutations.append("vertical")
        if agent_position[1] > n//2:
            agent_position_1 = n//2 - (agent_position[1] - (n//2))
            agent_position = (agent_position[0], agent_position_1)
            if coin_position != (0,0):
                coin_position_1 = n//2 - (coin_position[1] - (n//2))
                coin_position = (coin_position[0], coin_position_1)
            if include_bombs and bomb_position != (0,0):
                bomb_position_1 = n//2 - (bomb_position[1] - (n//2))
                bomb_position = (bomb_position[0], bomb_position_1)
            permutations.append("horizontal")
        if coin_position[1] > coin_position[0]:
            if coin_position != (0, 0):
                coin_position = (coin_position[1], coin_position[0])
            agent_position = (agent_position[1], agent_position[0])
            if include_bombs and bomb_position != (0, 0):
                bomb_position = (bomb_position[1], bomb_position[0])
            permutations.append("diagonal")

        number_of_agent_tiles = int(3*n**2/16)  # for the agent, only a part of the tiles are valid
        first_digit = convert_pos(coin_position, n, is_a_coin=True)  # no minus 1 here. The states are numbered 1 to
        # number_of_coin_tiles, and 0 is the state with no coin on the board.
        second_digit = convert_pos(agent_position, n//2+1) - 1  # There has to be the agent on the board, so the - 1 is
        # needed here. The numbering is from 0 to (number_of_agent_tiles-1)

        index = first_digit * number_of_agent_tiles + second_digit

        if include_bombs:
            number_of_coin_tiles = 1  # This one tile here is the tile (0,0), the state with no coin on the board
            for k in range(1, n):
                if k % 2 == 0:
                    number_of_coin_tiles += k // 2
                else:
                    number_of_coin_tiles += k

            third_digit = convert_pos(bomb_position, n)  # also here no minus one, because of the (0,0) position.
            number_of_agent_coin_states = number_of_coin_tiles*number_of_agent_tiles

            index += third_digit * number_of_agent_coin_states

        if include_crates:
            number_of_bomb_states = int((n - 1) ** 2 - (n / 2 - 1) ** 2) + 1  # this is the total number of valid positions
            old_crate_up = int(game_state['field'][agent_position[0], agent_position[1] - 1] == 1)
            old_crate_right = int(game_state['field'][agent_position[0] + 1, agent_position[1]] == 1)
            old_crate_down = int(game_state['field'][agent_position[0], agent_position[1] + 1] == 1)
            old_crate_left = int(game_state['field'][agent_position[0] - 1, agent_position[1]] == 1)

            # check if other agents are nearby (they are treated equal to crates)
            others_positions = [agent_attributes[-1] for agent_attributes in game_state['others']]  # gather the other
            # agents positions
            agent_x, agent_y = agent_position
            for (others_x, others_y) in others_positions:
                if others_x == agent_x and others_y + 1 == agent_y:
                    old_crate_up = 1
                elif others_x - 1 == agent_x and others_y == agent_y:
                    old_crate_right = 1
                elif others_x == agent_x and others_y - 1 == agent_y:
                    old_crate_down = 1
                elif others_x + 1 == agent_x and others_y == agent_y:
                    old_crate_left = 1


            crate_up, crate_right, crate_down, crate_left = crate_permutator([old_crate_up,
                                                                              old_crate_right,
                                                                              old_crate_down,
                                                                              old_crate_left],
                                                                             permutations)

            fourth_digit = crate_up + crate_right * 2 + crate_down * 4 + crate_left * 8  # binary number covering
            # combinations of crates located right next to the agent

            index += fourth_digit * number_of_agent_coin_states*number_of_bomb_states

        return index, permutations

    # This is only reached, if there is no dimension reduction.
    number_of_tiles = int(3/4 * n**2 - n)  # total number of tiles on the board
    first_digit = convert_pos(coin_position, n)  # number of the tile where the agent is located
    second_digit = convert_pos(agent_position, n) - 1  # number of the tile where the coin is located

    index = first_digit * number_of_tiles + second_digit  # a unique integer, representing the state
    if include_bombs:
        if len(custom_bomb_state):
            if bomb_index is None:  # if no bomb index is handed, find the index of the closest bomb
                bomb_index = index_of_closest_item(agent_position,
                                                   [custom_bomb_state[k][0] for k in range(len(custom_bomb_state))])

            bomb_position = custom_bomb_state[bomb_index][0]
        else:
            bomb_position = (0, 0)  # if there is no bomb (len(game_state[bombs])) == 0(False), position is set to (0,0)
        third_digit = convert_pos(bomb_position, n)  # number of the tile, where the bomb is located

        index += third_digit * number_of_tiles*(number_of_tiles+1)  # #number_of_tiles states for the agent and #number_
        # _of_tiles + 1 for the coin

    if include_crates:
        crate_up = int(game_state['field'][agent_position[0], agent_position[0] - 1] == 1)
        crate_right = int(game_state['field'][agent_position[0] + 1, agent_position[0]] == 1)
        crate_down = int(game_state['field'][agent_position[0], agent_position[0] + 1] == 1)
        crate_left = int(game_state['field'][agent_position[0] - 1, agent_position[0]] == 1)

        # check if other agents are nearby (they are treated equal to crates)
        others_positions = [agent_attributes[-1] for agent_attributes in game_state['others']]  # gather the other
        # agents positions
        agent_x, agent_y = agent_position
        for (others_x, others_y) in others_positions:
            if others_x == agent_x and others_y + 1 == agent_y:
                crate_up = 1
            elif others_x - 1 == agent_x and others_y == agent_y:
                crate_right = 1
            elif others_x == agent_x and others_y - 1 == agent_y:
                crate_down = 1
            elif others_x + 1 == agent_x and others_y == agent_y:
                crate_left = 1
                
        fourth_digit = crate_up + crate_right * 2 + crate_down * 4 + crate_left * 8
        index += fourth_digit * number_of_tiles * (number_of_tiles+1)**2
    return index, permutations

def crate_permutator(old_crates: list[int], permutations: list[str]) -> list[int]:
    index_permutations = {"diagonal": [3, 2, 1, 0],
                          "vertical": [0, 3, 2, 1],
                          "horizontal": [2, 1, 0, 3]
                          }
    current_index_set = [0, 1, 2, 3]
    for permutation in permutations:
        for i in range(len(current_index_set)):
            current_index_set[i] = index_permutations[permutation][current_index_set[i]]
    new_crates = [0, 0, 0, 0]
    for i in range(len(new_crates)):
        new_crates[current_index_set[i]] = old_crates[i]
    return new_crates




def distance_from_item(agent_position: tuple, item_position: tuple):
    item_x, item_y = item_position
    self_x, self_y = agent_position
    return abs(item_x - self_x) + abs(item_y - self_y)


def revert_permutations(action, permutations):
    """
    This function undoes the effect, that permutations have. Since all permutations are self_inverse, D*D = id,
    H*H = id, V*V = id, all we have to do is to apply the permutations in reversed order. The effect of each permutation
    is stored in the corresponding dictionary below.
    """
    vert_dict = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT", "BOMB": "BOMB"}
    horiz_dict = {"LEFT": "LEFT", "RIGHT": "RIGHT", "UP": "DOWN", "DOWN": "UP", "WAIT": "WAIT", "BOMB": "BOMB"}
    diag_dict = {"LEFT": "UP", "RIGHT": "DOWN", "UP": "LEFT", "DOWN": "RIGHT", "WAIT": "WAIT", "BOMB": "BOMB"}
    inverse_permutations = {"vertical": vert_dict, "horizontal": horiz_dict, "diagonal": diag_dict}
    new_action = action
    for k in range(len(permutations)):
        old_action = new_action
        permutation_to_invert = permutations[len(permutations) - (k+1)]
        new_action = inverse_permutations[permutation_to_invert][old_action]
    return new_action


def apply_permutations(action, permutations):
    """
    This function is somewhat the opposite of the revert_permutations function. It applies all permutations. Since
    revert_permutations does nothing other than applying the permutations in the reversed order, we can call it with a
    reversed input, to simply apply the permutations.
    """
    reversed_permutations = [permutations[len(permutations)-i-1] for i in range(len(permutations))]
    return revert_permutations(action, reversed_permutations)

def get_all_explosions(game_state):
    explosion_map = game_state['explosion_map']

    # no explosion currently on the map
    if np.sum(explosion_map) == 0.0:
        return []

    explosions = []
    for x in range(len(explosion_map)):
        for y in range(len(explosion_map)):
            tile = explosion_map[x][y]
            if tile == 1:
                explosions.append((x, y))

    return explosions

def in_danger(game_state):
    # todo
    pass






