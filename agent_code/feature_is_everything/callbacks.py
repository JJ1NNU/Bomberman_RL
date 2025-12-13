import pickle
import random
import torch

from .features import Feature

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def encode_feature(features):
    if features is None or len(features) == 0:
        return [0] * 34
    mapping_0_4 = {'block': 0, 'free': 1, 'dead': 2, 'coin': 3, 'enemy': 4, 'target': 5}
    mapping_5 = {'True': 0, 'False': 1, 'target': 2, 'KILL!': 3}
    
    onehot_vector = []
    
    for i in range(5):
        onehot = [0] * 6
        value = features[i]
        if value in mapping_0_4:
            idx = mapping_0_4[value]
            onehot[idx] = 1
        onehot_vector.extend(onehot)
    
    onehot = [0] * 4
    value = features[5]
    if value in mapping_5:
        idx = mapping_5[value]
        onehot[idx] = 1
    onehot_vector.extend(onehot)
    
    return onehot_vector


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
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.feature = Feature()
    if not self.train:
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file).to(self.device)
        self.logger.info("Loading model from saved state.")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = self.feature(game_state)

    features_tensor = torch.tensor(encode_feature(features), dtype=torch.float32).unsqueeze(0).to(self.device)
    if self.train:
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(features_tensor)
                action_index = q_values.max(1)[1].item()
        else:
            action_index = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            q_values = self.policy_net(features_tensor)
            self.logger.debug(f"q_values: {q_values}")
            action_index = q_values.max(1)[1].item()

    # If train with itself
    if game_state["round"] % 500 == 0 and game_state["step"] == 1:
        setup(self)
    if self.train:
        self.steps_done += 1

    return ACTIONS[action_index]
