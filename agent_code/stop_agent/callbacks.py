import numpy as np

def setup(self):
    np.random.seed()
    pass

def act(self, game_state: dict):
    return 'WAIT'
