import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device('cpu')

def setup(self):
    self.coordinate_history = deque(maxlen=20)
    self.action_history = deque(maxlen=20)
    self.bomb_cooldown = 0
    self.step = 0
    self.policy = None
    
    # 🔑 학습/플레이 자동 감지
    self.is_training = os.environ.get('GAIL_TRAINING', '0') == '1'
    
    if os.path.isfile("my-saved-model.pt"):
        try:
            checkpoint = torch.load("my-saved-model.pt", map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                self.policy_dict = checkpoint
            else:
                self.policy = checkpoint
                if hasattr(self.policy, 'eval'):
                    self.policy.eval()
            self.logger.debug("✅ GAIL PPO loaded!")
        except Exception as e:
            self.logger.error(f"Model load failed: {e}")

def state_to_features(self, game_state) -> np.ndarray:
    if game_state is None:
        return np.zeros((4, 7, 7), dtype=np.float32)
    
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    others = game_state["others"]
    my_pos = game_state["self"][3]

    reach = 3
    h, w = 7, 7
    vision_coordinates = np.indices((h, w))
    vision_coordinates[0] += my_pos[0] - reach
    vision_coordinates[1] += my_pos[1] - reach
    vision_coordinates = vision_coordinates.T.reshape((-1, 2))

    wall_crates = np.full((h, w), -1.0)
    explosion_coins = np.zeros((h, w))
    bomb_enemies = np.zeros((h, w))
    empty = np.zeros((h, w))

    # Wall/Crates
    for coord in vision_coordinates:
        if 0 < coord[0] < field.shape[0] and 0 < coord[1] < field.shape[1]:
            x, y = int(coord[0] - my_pos[0] + reach), int(coord[1] - my_pos[1] + reach)
            wall_crates[y, x] = field[int(coord[0]), int(coord[1])]

    # Explosion/Coins
    for i, j in np.transpose((explosion_map > 0).nonzero()):
        if any((i == c[0] and j == c[1]) for c in vision_coordinates):
            x, y = int(i - my_pos[0] + reach), int(j - my_pos[1] + reach)
            if 0 <= x < h and 0 <= y < w:
                explosion_coins[y, x] = -1
    for coin in coins:
        if any((coin[0] == c[0] and coin[1] == c[1]) for c in vision_coordinates):
            x, y = int(coin[0] - my_pos[0] + reach), int(coin[1] - my_pos[1] + reach)
            if 0 <= x < h and 0 <= y < w:
                explosion_coins[y, x] = 1

    # Bomb/Enemies
    for enemy in others:
        pos = enemy[3]
        if any((pos[0] == c[0] and pos[1] == c[1]) for c in vision_coordinates):
            x, y = int(pos[0] - my_pos[0] + reach), int(pos[1] - my_pos[1] + reach)
            if 0 <= x < h and 0 <= y < w:
                bomb_enemies[y, x] = 1
    for bomb in bombs:
        pos = bomb[0]
        if any((pos[0] == c[0] and pos[1] == c[1]) for c in vision_coordinates):
            x, y = int(pos[0] - my_pos[0] + reach), int(pos[1] - my_pos[1] + reach)
            if 0 <= x < h and 0 <= y < w:
                bomb_enemies[y, x] = -1

    features = np.stack([wall_crates, explosion_coins, bomb_enemies, empty], axis=0)
    return features.astype(np.float32)

def act(self, game_state: dict) -> str:
    self.step = game_state['step']
    self.x, self.y = game_state['self'][3]

    if self.step == 1:
        self.coordinate_history.clear()
        self.action_history.clear()
    self.coordinate_history.append((self.x, self.y))

    raw_action = _choose_gail_action(self, game_state)

    # 🔑 학습 모드: shielding 완전 OFF
    if self.is_training:
        self.action_history.append(raw_action)
        return raw_action

    # 플레이 모드: Safety Shield ON
    try:
        safe_actions = get_filtered_actions(game_state, self.action_history)
    except:
        safe_actions = ['WAIT']
    
    final_action = raw_action
    if raw_action not in safe_actions:
        candidate_actions = [a for a in safe_actions if a != 'WAIT']
        final_action = np.random.choice(candidate_actions) if candidate_actions else np.random.choice(safe_actions)
        self.logger.debug(f"🛡️ Shield: {raw_action} → {final_action}")
    
    self.action_history.append(final_action)
    return final_action

def _choose_gail_action(self, game_state):
    obs = state_to_features(self, game_state)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    
    if hasattr(self, 'policy') and self.policy is not None:
        try:
            with torch.no_grad():
                action_logits = self.policy(obs_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()
            return ACTIONS[action_idx]
        except:
            pass
    
    return np.random.choice(ACTIONS)
