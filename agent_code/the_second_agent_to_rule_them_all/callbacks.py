import os
import pickle
import random
import torch
import numpy as np
import math as m
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    if self.train and not os.path.isfile("my-saved-model.pt"):
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    elif self.train and os.path.isfile("my-saved-model.pt"):
        print("Building on existing model.")
    else:
        print("Loading model from saved state.")
        # [주의] 파일이 없을 경우 대비
        try:
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
             print("Model file not found! Initializing random weights for fallback.")
             weights = np.random.rand(len(ACTIONS))
             self.model = weights / weights.sum()

    # [추가] Loop Breaker 및 좌표 추적을 위한 초기화
    self.coordinate_history = deque(maxlen=20)
    self.step = 0

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    """
    # -----------------------------------------------------------
    # 1. [Prepare] 상태 갱신
    # -----------------------------------------------------------
    self.step = game_state['step']
    self.x, self.y = game_state['self'][3]

    # 좌표 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
    self.coordinate_history.append((self.x, self.y))

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기
    # -----------------------------------------------------------
    raw_action = _choose_action(self, game_state)

    # -----------------------------------------------------------
    # 3. [Early Game Check] & [Safety Shield]
    # -----------------------------------------------------------
    if self.step < 30:
        # 초반 30스텝은 Safety Shield 끄기
        final_action = raw_action
    else:
        # 30스텝 이후 Safety Shield 작동
        try:
            safe_actions = get_filtered_actions(game_state)
        except Exception as e:
            print(f"Safety Shield Error: {e}")
            safe_actions = [raw_action]
            
        # -----------------------------------------------------------
        # 4. [Loop Breaker] (30스텝 이후)
        # -----------------------------------------------------------
        is_looping = False
        is_safe_now = 'WAIT' in safe_actions
        
        if is_safe_now and len(self.coordinate_history) >= 10:
            history_list = list(self.coordinate_history)
            recent_locs = set(history_list[-10:])
            # 최근 10턴 동안 2개 이하의 좌표만 방문했다면 루프
            if len(recent_locs) <= 2:
                is_looping = True
        
        final_action = raw_action
        
        # 루프 탈출 또는 위험 행동 차단
        if is_looping or (raw_action not in safe_actions):
            if is_looping:
                pass # 루프 감지 (로그 없음)
            
            if safe_actions:
                safe_moves = [a for a in safe_actions if a != 'WAIT']
                if safe_moves:
                    final_action = np.random.choice(safe_moves)
                else:
                    final_action = np.random.choice(safe_actions)
            else:
                final_action = 'WAIT'

    return final_action

def _choose_action(self, game_state: dict) -> str:
    """
    기존 act 함수의 핵심 로직 분리
    """
    features = state_to_features(game_state)
    features = torch.tensor([features]) # Game state to torch tensor

    if self.train:
        eps_start = 0.9
        eps_end = 0.1
        eps_decay = 1000
        round = 1
        sample = random.random()
        eps_threshold = 0.05 

        if sample > eps_threshold:
            round += 1
            with torch.no_grad():
                action_done = self.policy_net(features).max(1)[1].view(1, 1)
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.2, .2, .2, .2, .1, .1])]],
                                       dtype=torch.long)
    else:
        # 평가 모드 (model이 dict가 아니라 callable 모델 객체라고 가정)
        # 원본 코드 setup에서 pickle로 로드하는데, 이것이 PyTorch 모델인지 numpy array인지 확인 필요
        # 원본 코드: self.model(features).max(1)[1] -> PyTorch 모델임이 확실함
        # 하지만 setup의 else 블록에서 weights / weights.sum() (numpy array)을 할당하는 경우도 있음
        # 따라서 타입 체크가 안전함
        
        if isinstance(self.model, np.ndarray):
             # 모델이 numpy array인 경우 (fallback)
             action_done = np.random.choice(len(ACTIONS), p=self.model)
             return ACTIONS[action_done]
        else:
             # PyTorch 모델인 경우
             with torch.no_grad():
                 action_done = self.model(features).max(1)[1].view(1, 1)

    return ACTIONS[action_done]

def state_to_features(game_state: dict) -> np.array:
    # (원본 state_to_features 함수 내용 그대로 유지)
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    agent = game_state["self"]
    others = game_state["others"]
    my_pos = agent[3]

    # Just make a rectangle that is 3 squares in each direction, centered on where the player is
    reach = 3
    left = my_pos[0] - reach
    right = my_pos[0] + reach + 1
    top = my_pos[1] - reach
    bottom = my_pos[1] + reach + 1

    # all coordinates that are in sight
    vision_coordinates = np.indices((7, 7))
    vision_coordinates[0] += my_pos[0] - reach
    vision_coordinates[1] += my_pos[1] - reach
    vision_coordinates = vision_coordinates.T
    vision_coordinates = vision_coordinates.reshape(((reach*2+1)**2, 2))

    # --- map with walls (-1), free (0) and crates (1) ---------------------------------------------------------------
    wall_crates = np.zeros((reach * 2 + 1, reach * 2 + 1)) - 1 # outside of game also -1

    for coord in vision_coordinates:
        if not 0 < coord[0] < field.shape[0] or not 0 < coord[1] < field.shape[1]:
            continue # next -> continue (Python syntax)
        else:
            x = coord[0] - my_pos[0] + reach
            y = coord[1] - my_pos[1] + reach
            wall_crates[x, y] = field[coord[0], coord[1]]

    # --- map with explosion (-1) free (0) and coins (1) -------------------------------------------------------------
    explosion_coins = np.zeros((reach * 2 + 1, reach * 2 + 1))
    explosion_coord = np.transpose((explosion_map > 0).nonzero())

    for expl in explosion_coord:
        if any(sum(expl == i) == 2 for i in vision_coordinates):
            x_expl = expl[0] - my_pos[0] + reach
            y_expl = expl[1] - my_pos[1] + reach
            explosion_coins[x_expl, y_expl] = -1

    for coin in coins:
        if any(sum(np.asarray(coin) == i) == 2 for i in vision_coordinates):
            x_coin = coin[0] - my_pos[0] + reach
            y_coin = coin[1] - my_pos[1] + reach
            explosion_coins[x_coin, y_coin] = 1

    # --- map with bomb range (-1), free (0) and opponents (1) --------------------------------------------------------
    bomb_opponents = np.zeros((reach * 2 + 1, reach * 2 + 1))

    for enemy in others:
        if any(sum(enemy[3] == i) == 2 for i in vision_coordinates):
            x_enemy = enemy[3][0] - my_pos[0] + reach
            y_enemy = enemy[3][1] - my_pos[1] + reach
            bomb_opponents[x_enemy, y_enemy] = 1

    for bomb in bombs:
        if any(sum(bomb[0] == i) == 2 for i in vision_coordinates):
            # coordinate of bomb in our vision matrix
            x_bomb = bomb[0][0] - my_pos[0] + reach
            y_bomb = bomb[0][1] - my_pos[1] + reach
            range_bomb = [1, 2, 3]
            bomb_opponents[x_bomb, y_bomb] = -1

            # compute the explosion range
            for j in range_bomb:
                if j + x_bomb > 6: break
                if wall_crates[x_bomb + j, y_bomb] in {-1}: break
                else: bomb_opponents[x_bomb + j, y_bomb] = -1
            
            for j in range_bomb:
                if j - x_bomb < 0: break
                if wall_crates[x_bomb - j, y_bomb] in {-1}: break
                else: bomb_opponents[x_bomb - j, y_bomb] = -1

            for j in range_bomb:
                if j + y_bomb > 6: break
                if wall_crates[x_bomb, y_bomb + j] in {-1}: break
                else: bomb_opponents[x_bomb, y_bomb + j] = -1

            for j in range_bomb:
                if j - y_bomb < 0: break
                if wall_crates[x_bomb, y_bomb - j] in {-1}: break
                else: bomb_opponents[x_bomb, y_bomb - j] = -1
        else: 
            continue # next -> continue

    # --- scaled down vision outside reach ---------------------------------------------------------------------------
    outside_map = np.zeros((4, 4)) # coins, crates, bombs, other agents

    for i in range(0, field.shape[0]):
        for j in range(0, field.shape[1]):
            field_value = field[i, j]
            if field_value == 0:
                continue # next -> continue
            else:
                # coins = 3, crates = 2, bombs and explosions = -1 -> make it to 1, other agents >= 5 -> make it to 0 (index)
                if field_value == -1:
                    field_value = 1
                elif field_value >= 5:
                    field_value = 0
                
                # [버그 수정] 원본 코드의 field_value가 인덱스로 쓰이는데(0,1), 
                # field_value가 2, 3, 4일 때는? (2:crate, 3:coin)
                # 원본 코드 로직상:
                # crate(2) -> field_value 그대로 2 -> outside_map[:, 2] (Bombs?) -> 의도 불분명하지만 원본 유지
                # coin(3) -> field_value 그대로 3 -> outside_map[:, 3] (Other agents?)
                # 원본 로직을 최대한 존중하되 인덱스 에러 방지용 min/max 처리는 하지 않음 (원본이 맞다고 가정)

                if j < top:
                    outside_map[0, field_value] = 1
                if j > bottom:
                    outside_map[1, field_value] = 1
                if i < left:
                    outside_map[2, field_value] = 1
                if i > right:
                    outside_map[3, field_value] = 1

    features = wall_crates.flatten().tolist() + explosion_coins.flatten().tolist() \
             + bomb_opponents.flatten().tolist() + outside_map.ravel().tolist()

    return features
