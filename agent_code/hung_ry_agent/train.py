from typing import List
import numpy as np
import matplotlib.pyplot as plt

import events as e
from ..base_agent import main as base
from ..base_agent.main import ACTIONS
from ..base_agent import train as base_train
from .features import get_features, get_tile_type, count_nearby_crates, count_nearby_opponents

# Events
CLOSER_TO_COIN = "closer_to_coin"
CLOSER_TO_CRATE = "closer_to_crate"
BOMB_SCORE = "bomb_score"
TARGETED_OPPONENTS = "targeted_opponents"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    base_train.setup_training(self)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if self.eval_mode:
        base_train.collect_data(self, events)
        return

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_game_state, transformation = base.get_standarized_state_dict(old_game_state)
    new_game_state, _ = base.get_standarized_state_dict(new_game_state)
    self_action = base.forward_move_transformations(self_action, transformation)

    old_features, new_features, old_id, new_id = base.store_state(self, old_game_state, new_game_state, get_features)

    self.new_features = new_features

    if e.KILLED_OPPONENT in events:
        print("KILLED OPPONENT")
    elif e.OPPONENT_ELIMINATED in events:
        print("OPPONENT DIED")


    events.append(CLOSER_TO_COIN)
    events.append(CLOSER_TO_CRATE)
    events.append(BOMB_SCORE)
    if e.BOMB_DROPPED in events:
        events.append(TARGETED_OPPONENTS) 
        

    bomb_score, targeted_opponents = calculate_bomb_score(self, old_game_state, new_game_state, self_action == "BOMB")
    coin_score = moved_to_coin(self, old_features[1], new_features[1], e.COIN_COLLECTED in events)
    crate_score = moved_to_crate(self, old_features[6], self_action)

    # print('-------------')
    # print("Step: ", old_game_state["step"])
    # print("Old user coords: ", old_game_state["self"][3])
    # print("Placed bomb: ", self_action == "BOMB")
    # print("Action: ", self_action)
    # print("Bomb score: ", bomb_score)
    # print("Targeted opponents: ", targeted_opponents)
    # print("Coins: ", coin_score)
    # print("Crates: ", crate_score)
    # print("Progress: ", progress)


    custom_rewards = {
        CLOSER_TO_COIN: coin_score,
        CLOSER_TO_CRATE: crate_score,
        BOMB_SCORE: bomb_score,
        TARGETED_OPPONENTS: targeted_opponents
    }

    reward = reward_from_events(self, events, custom_rewards)

    action_id = ACTIONS.index(self_action)


    use_SARSA = True
    if use_SARSA:
        if len(self.stored_transition) > 0:
            SARSA = self.stored_transition + [action_id]
            base_train.update_q_table_SARSA(self, SARSA)
        self.stored_transition = [old_id, action_id, reward, new_id]
    
    else:
        base_train.update_q_table(self, old_id, action_id, new_id, reward)




def reward_from_events(self, events: List[str], custom_rewards) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """ 
    
    game_rewards = {
        e.COIN_COLLECTED: 15,
        CLOSER_TO_COIN: custom_rewards[CLOSER_TO_COIN], # Got closer to a coin
        CLOSER_TO_CRATE: custom_rewards[CLOSER_TO_CRATE], # Got closer to a crate
        TARGETED_OPPONENTS: custom_rewards[TARGETED_OPPONENTS],
        BOMB_SCORE: custom_rewards[BOMB_SCORE], # Quality of the placed bomb
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            # print(f"Awarded {game_rewards[event]} for event {event}")

    # print(f"Awarded {reward_sum} in total")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    base_train.end_of_round(self, last_game_state, last_action, events)




def moved_to_coin(self, old_coin_features, new_coin_features, coin_collected):
    """
    Checks if the agent moved closer to a coin.
    Returns 1 if it did, 0 if it stayed the same and -1 if it moved further away.
    If the coin was collected by the agent, returns 0.
    If no coin was present, returns 0.
    If the coin is too far away, returns 0.
    """
    old_coin_distance = np.abs(old_coin_features).sum()
    new_coin_distance = np.abs(new_coin_features).sum()

    if old_coin_distance > 10 and new_coin_distance > 10:
        return 0

    if new_coin_distance > old_coin_distance:
        if coin_collected:
            return 0
        else:
            return -1
    elif new_coin_distance == old_coin_distance:
        return 0
    else:
        return 4


def moved_to_crate(self, old_direction: np.ndarray, action: str) -> int:
    """
    Checks if the agent moved closer the dense spot.
    Returns 1 if it did, 0 if it stayed the same and -1 if it moved further away.
    If no crate was present, returns 0.
    """

    if action == "BOMB" or action == "WAIT":
        return 0
    
    good_actions = []
    neutral_actions = []

    if old_direction[1] == -1:
        good_actions.append("UP")
    elif old_direction[1] == 1:
        good_actions.append("DOWN")
    elif old_direction[1] == 0:
        neutral_actions += ["UP", "DOWN"]

    if old_direction[0] == -1:
        good_actions.append("LEFT")
    elif old_direction[0] == 1:
        good_actions.append("RIGHT")
    elif old_direction[0] == 0:
        neutral_actions += ["LEFT", "RIGHT"]

    if action in good_actions:
        return 1
    if action in neutral_actions:
        return 0
    return -1
    
def estimate_bomb_effectiveness(field: np.ndarray, user_coords: list, others: list = []) -> tuple[int, int]:
    """
    Returns the score of placing a bomb in the current user position.
    Rewards placing bombs close to crates and opponents.
    Rewards placing bombs in intersections.
    Penalizes placing bombs in corners.
    Penalizes placing bombs in empty spaces.
    """

    user_coords = np.array(user_coords)
    opponent_coords = np.array([other[3] for other in others])

    _, blast_mask = get_tile_type(field, user_coords)
    tiles_hit = blast_mask.sum()
    crates_hit = count_nearby_crates(field, user_coords, blast_mask)
    opponents_targeted = count_nearby_opponents(opponent_coords, user_coords)

    if crates_hit == 0 and opponents_targeted == 0:
        score = -3
    else:
        score = (tiles_hit - 6)/2 + 3 * crates_hit

    return score, 3 * opponents_targeted


def calculate_bomb_score(self, old_game_state, new_game_state, placed_bomb: bool) -> tuple[int, int]:
    """
    Rewards for moving to a tile with a higher bomb score.
    """
    had_bomb = old_game_state["self"][2]
    old_bomb_effectiveness, opponents_targeted  = estimate_bomb_effectiveness(old_game_state["field"], old_game_state["self"][3], old_game_state["others"])
    new_bomb_effectiveness, _ = estimate_bomb_effectiveness(new_game_state["field"], new_game_state["self"][3], new_game_state["others"])
    
    user_coords = old_game_state["self"][3]

    valid_moves = base.get_valid_moves(
        old_game_state["field"],
        user_coords,
        bomb_possible=True,
        allow_wait=True,
    )

    safe_moves = base.get_safe_moves(old_game_state, valid_moves)

    is_new_bombing_safe = base.get_safe_moves(new_game_state, [False, False, False, False, False, True])[5]

    if not safe_moves[5]:
        old_bomb_effectiveness = 0
    
    if not is_new_bombing_safe:
        new_bomb_effectiveness = 0

    if old_bomb_effectiveness == 0 and new_bomb_effectiveness == 0:
        return 0, opponents_targeted

    if not placed_bomb:
        if new_bomb_effectiveness < old_bomb_effectiveness:
            return (-3 if had_bomb else -1), opponents_targeted
        if new_bomb_effectiveness == old_bomb_effectiveness and had_bomb:
            return -1, opponents_targeted

    new_positions = [
        [user_coords[0], user_coords[1] - 1], # Up
        [user_coords[0] + 1, user_coords[1]], # Right
        [user_coords[0], user_coords[1] + 1], # Down
        [user_coords[0] - 1, user_coords[1]] # Left
    ]

    old_adjacent_tiles = [new_positions[i] for i in range(4) if (valid_moves[i] and base.get_safe_moves({**old_game_state, "self": (0, 0, False, new_positions[i])}, [False, False, False, False, False, True])[5])]

    if (safe_moves == True).sum() < 2: # Only 2 or less valid moves 
        return 0, opponents_targeted
    old_adjacent_scores = [estimate_bomb_effectiveness(old_game_state["field"], tile, old_game_state["others"])[0] for tile in old_adjacent_tiles]
    max_adjacent_score = 0 if len(old_adjacent_scores) == 0 else max(old_adjacent_scores)
    if placed_bomb:
        if old_bomb_effectiveness >= max_adjacent_score: # Placed bomb in a local maximum
            return old_bomb_effectiveness, opponents_targeted
        return -1, opponents_targeted
    if new_bomb_effectiveness >= max_adjacent_score: # Moved to a tile with the highest bomb score
        return 2, opponents_targeted
    if new_bomb_effectiveness > old_bomb_effectiveness: # Moved to a tile with a higher bomb score
        return 1, opponents_targeted
    else:
        return 0, opponents_targeted