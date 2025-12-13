import numpy as np

from ..base_agent import main as base
from ..base_agent import features as base_features
from .features import get_features

 

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
    self.eval_mode = True
    base.setup(self, initialize, resume_training=False)
    self.weights = np.array([0.1, 10, 1, 1, 1, 10, 1])


def initialize(self):
    self.features = [
        np.zeros((0, 1)), # Step
        np.zeros((0, 2)), # Closest coin
        np.zeros((0, 1)), # Nearby crates
        np.zeros((0, 2)), # Closest opponent
        np.zeros((0, 1)), # Nearby opponents
        #np.zeros((0, 4, 2)), # Bombs
        np.zeros((0, 5)), # tile type
        np.zeros((0, 2)) # Direction of dense crates
    ]
    self.Q_table = np.zeros((0,6))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    game_state, transformations = base.get_standarized_state_dict(game_state)


    field = game_state['field'].copy()
    field = base.add_coords_to_matrix(field, np.array([other[3] for other in game_state['others']]), value=1)
    field = base.add_coords_to_matrix(field, np.array([bomb[0] for bomb in game_state['bombs']]), value=1)

    valid_moves = base.get_valid_moves(
        field, 
        game_state['self'][3], 
        bomb_possible=game_state['self'][2], 
        allow_wait=True
    )

    safe_moves = base.get_safe_moves(game_state, valid_moves)
    # print("--------------------")

    play_random = False
    if play_random:
        selected_move = base.select_move(np.random.random(6), safe_moves)
    else:
        if self.train and not self.eval_mode and np.random.random() < base.epsion_greedy(self, initial_eps=0.9, final_eps=0.1, decay=0.005):
            selected_move = base.select_move(np.random.random(6), safe_moves)
        else:
            features = self.new_features if (self.train and not self.eval_mode and game_state['step'] != 1) else get_features(game_state)

            idx, is_known = base_features.get_state_id(self, features)
            
            if is_known:
                not_visited = self.Q_table[idx] == 0
                need_regression = (not_visited * safe_moves).any()

                if need_regression:
                    regression_values = base.Q_regression(self, features, is_known)
                    # print("Regression values: ", regression_values)
                    # print("But some are known: ", self.Q_table[idx])
                    regression_values = self.Q_table[idx] * (1 - not_visited) + regression_values * not_visited
                    #regression_values = (0.75*self.Q_table[idx] + 0.25*regression_values) * (1 - not_visited) + regression_values * not_visited


                    selected_move = base.select_move(regression_values, safe_moves)
                else:
                    # print("Known values: ", self.Q_table[idx])
                    selected_move = base.select_move(self.Q_table[idx], safe_moves)
            
            else:
                regression_values = base.Q_regression(self, features, is_known)
                # print("Regression values: ", regression_values)
                selected_move = base.select_move(regression_values, safe_moves)

    #print("Selected move: ", selected_move)
    #print("Transformed move: ", base.revert_move_transformations(selected_move, transformations))
    return base.revert_move_transformations(selected_move, transformations)

