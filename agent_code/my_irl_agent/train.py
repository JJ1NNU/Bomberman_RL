from collections import namedtuple, deque
import pickle
import numpy as np
from typing import List

import events as e

# GAIL 학습은 외부(train_gail.py)에서 진행되므로, 
# 여기서는 아무 동작도 하지 않도록(pass) 빈 함수로 정의만 해둡니다.

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    """
    pass

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    pass

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    pass
