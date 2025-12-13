from typing import List
from collections import deque
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import ACTIONS, encode_feature
from .symmetry import adjust_action, adjust_state
from .model import DQN

# Events
MOVE_TO_DEAD = "MOVE_TO_DEAD"
MOVE_TO_TARGET = "MOVE_TO_TARGET"
ATTACK_TARGET = "ATTACK_TARGET"
ATTACK_ENEMY = "ATTACK_ENEMY"
KILL_ENEMY = "KILL_ENEMY"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.epsilon = 0.5
    self.epsilon_min = 0.1
    self.epsilon_decay = 0.9995
    self.gamma = 0.9
    self.batch_size = 64
    self.memory = deque(maxlen=10000)
    self.steps_done = 0
    self.update_target_steps = 1000  # How often to update the target network

    input_dim = 34 # 5 * 6 + 4
    output_dim = len(ACTIONS)
    try:
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file).to(self.device)
    except FileNotFoundError:
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
    self.target_net = DQN(input_dim, output_dim).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-3)
    self.loss_fn = nn.MSELoss()


def store_symmetric_transitions(self, state, action, reward, next_state, done):
    rotations = [0, 90, 180, 270]
    flips = [(False, False), (True, False), (False, True)]
    for rotation in rotations:
        for flip_horizontal, flip_vertical in flips:
            adjusted_state = adjust_state(state, rotation, flip_horizontal, flip_vertical)
            adjusted_action = adjust_action(action, rotation, flip_horizontal, flip_vertical)
            adjusted_next_state = [] if next_state is None else adjust_state(next_state, rotation, flip_horizontal, flip_vertical)
            adjusted_action_index = ACTIONS.index(adjusted_action)
            
            self.memory.append((torch.tensor(encode_feature(adjusted_state), dtype=torch.float32), adjusted_action_index, reward, torch.tensor(encode_feature(adjusted_next_state), dtype=torch.float32), done))


def sample_batch(self):
    batch = random.sample(self.memory, self.batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    state_batch = torch.stack(state_batch).to(self.device)
    action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(self.device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), dtype=torch.bool).to(self.device)
    if any(non_final_mask):
        non_final_next_states = torch.stack([s for s in next_state_batch if s is not None]).to(self.device)
    else:
        non_final_next_states = torch.empty((0, state_batch.shape[1])).to(self.device)
    done_batch = torch.tensor(done_batch, dtype=torch.bool).unsqueeze(1).to(self.device)
    return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, done_batch


def optimize_model(self):
    if len(self.memory) < self.batch_size:
        return

    state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, done_batch = sample_batch(self)
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(self.batch_size, 1).to(self.device)
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)

    expected_state_action_values = (next_state_values * self.gamma) * (~done_batch) + reward_batch

    loss = self.loss_fn(state_action_values, expected_state_action_values)

    # Opit.
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # update target net
    if self.steps_done % self.update_target_steps == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_feature = self.feature(old_game_state)
    new_feature = self.feature(new_game_state) if new_game_state else None

    if old_feature[ACTIONS.index(self_action)] == "target":
        events.append(MOVE_TO_TARGET)
    if old_feature[ACTIONS.index(self_action)] == "dead":
        events.append(MOVE_TO_DEAD)
    if self_action == 'BOMB':
        if 'enemy' in old_feature:
            events.append(ATTACK_ENEMY)
        if old_feature[-1] == 'target':
            events.append(ATTACK_TARGET)
        if old_feature[-1] == 'KILL!':
            events.append(KILL_ENEMY)
    
    reward = reward_from_events(self, events)
    done = False if new_game_state else True
    store_symmetric_transitions(self, old_feature, self_action, reward, new_feature, done)
    optimize_model(self)
    

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    old_feature = self.feature(last_game_state)
    new_feature = None

    if old_feature[ACTIONS.index(last_action)] == "target":
        events.append(MOVE_TO_TARGET)
    if old_feature[ACTIONS.index(last_action)] == "dead":
        events.append(MOVE_TO_DEAD)
    if last_action == 'BOMB':
        if 'enemy' in old_feature:
            events.append(ATTACK_ENEMY)
        if old_feature[-1] == 'target':
            events.append(ATTACK_TARGET)
        if old_feature[-1] == 'KILL!':
            events.append(KILL_ENEMY)

    reward = reward_from_events(self, events)
    done = True

    store_symmetric_transitions(self, old_feature, last_action, reward, new_feature, done)

    optimize_model(self)

    if self.epsilon >= self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    if last_game_state['round'] % 100 == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -5,
        e.BOMB_DROPPED: -5,
        e.INVALID_ACTION: -20,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -300,
        e.GOT_KILLED: -100.,
        MOVE_TO_DEAD: -100,
        MOVE_TO_TARGET: 50,
        ATTACK_TARGET: 50,
        ATTACK_ENEMY: 20,
        KILL_ENEMY: 500,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
