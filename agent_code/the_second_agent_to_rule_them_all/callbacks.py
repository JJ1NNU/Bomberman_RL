import os
import pickle
import random
import torch
import numpy as np
import math as m
import sys
from collections import deque

# [추가] 상위 경로 추가 (action_prune.py 접근용)
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

    # [추가] Loop Breaker 및 좌표/행동 추적을 위한 초기화
    self.coordinate_history = deque(maxlen=20)
    self.action_history = deque(maxlen=20) 
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

    # 좌표 및 행동 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
        self.action_history.clear()
        
    self.coordinate_history.append((self.x, self.y))

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기
    # -----------------------------------------------------------
    raw_action = _choose_action(self, game_state)

    # [★ 추가] 상자 옆에서 멍때림 방지 (Deadlock Breaking with Bomb)
    if game_state['self'][2] == True:
        # 최근 5턴간 폭탄을 놓지 않음
        recent_actions = list(self.action_history)[-5:]
        if len(recent_actions) >= 5 and all(a != 'BOMB' for a in recent_actions):
            # 상자 확인
            x, y = game_state['self'][3]
            field = game_state['field']
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            crate_nearby = any(field[nx, ny] == 1 for nx, ny in neighbors if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1])
            
            if crate_nearby:
                try:
                    # 안전 확인 (팀킬/자폭 방지)
                    temp_safe_actions = get_filtered_actions(game_state, self.action_history)
                    if 'BOMB' in temp_safe_actions:
                        # print("Force BOMB placement: Stuck next to crate & Safe to bomb.")
                        raw_action = 'BOMB'
                except Exception as e:
                    print(f"Bomb Check Error: {e}")

    # -----------------------------------------------------------
    # 3. [Safety Shield] & [Loop Breaker]
    # -----------------------------------------------------------
    # [수정] 30스텝 조건 제거 -> 항상 쉴드 켜짐
    try:
        # [수정] action_prune에 action_history 전달
        safe_actions = get_filtered_actions(game_state, self.action_history)
    except Exception as e:
        print(f"Safety Shield Error: {e}")
        safe_actions = ['WAIT']
        
    final_action = raw_action
    
    # 쉴드에 의해 원래 의도가 차단되었는지 확인
    if raw_action not in safe_actions:
        
        # [수정] 이동 의도였다면 폭탄 제외
        if raw_action != 'BOMB':
            candidate_actions = [a for a in safe_actions if a != 'BOMB' and a != 'WAIT']
        else:
            candidate_actions = [a for a in safe_actions if a != 'WAIT']

        # 후보 선택
        if candidate_actions:
            final_action = np.random.choice(candidate_actions)
        elif 'WAIT' in safe_actions:
            final_action = 'WAIT'
        elif safe_actions: 
            final_action = np.random.choice(safe_actions)
        else:
            final_action = 'WAIT'

    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)

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
                # policy_net이 정의되어 있지 않을 수 있으므로 체크 (원본은 self.model일 가능성 높음)
                if hasattr(self, 'policy_net'):
                    action_done = self.policy_net(features).max(1)[1].view(1, 1)
                elif hasattr(self, 'model') and callable(self.model):
                    action_done = self.model(features).max(1)[1].view(1, 1)
                else: # Fallback to numpy array model
                     action_done = torch.tensor([[np.random.choice(len(ACTIONS))]], dtype=torch.long)
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.2, .2, .2, .2, .1, .1])]],
                                        dtype=torch.long)
    else:
        # 평가 모드
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
    # (원본 state_to_features 함수 내용 그대로 복사)
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    agent = game_state["self"]
    others = game_state["others"]
    my_pos = agent[3]

    reach = 3
    left = my_pos[0] - reach
    right = my_pos[0] + reach + 1
    top = my_pos[1] - reach
    bottom = my_pos[1] + reach + 1

    vision_coordinates = np.indices((7, 7))
    vision_coordinates[0] += my_pos[0] - reach
    vision_coordinates[1] += my_pos[1] - reach
    vision_coordinates = vision_coordinates.T
    vision_coordinates = vision_coordinates.reshape(((reach*2+1)**2, 2))

    wall_crates = np.zeros((reach * 2 + 1, reach * 2 + 1)) - 1 

    for coord in vision_coordinates:
        if not 0 < coord[0] < field.shape[0] or not 0 < coord[1] < field.shape[1]:
            continue 
        else:
            x = coord[0] - my_pos[0] + reach
            y = coord[1] - my_pos[1] + reach
            wall_crates[x, y] = field[coord[0], coord[1]]

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

    bomb_opponents = np.zeros((reach * 2 + 1, reach * 2 + 1))

    for enemy in others:
        if any(sum(enemy[3] == i) == 2 for i in vision_coordinates):
            x_enemy = enemy[3][0] - my_pos[0] + reach
            y_enemy = enemy[3][1] - my_pos[1] + reach
            bomb_opponents[x_enemy, y_enemy] = 1

    for bomb in bombs:
        if any(sum(bomb[0] == i) == 2 for i in vision_coordinates):
            x_bomb = bomb[0][0] - my_pos[0] + reach
            y_bomb = bomb[0][1] - my_pos[1] + reach
            range_bomb = [1, 2, 3]
            bomb_opponents[x_bomb, y_bomb] = -1

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
            continue 

    outside_map = np.zeros((4, 4)) 

    for i in range(0, field.shape[0]):
        for j in range(0, field.shape[1]):
            field_value = field[i, j]
            if field_value == 0:
                continue 
            else:
                if field_value == -1:
                    field_value = 1
                elif field_value >= 5:
                    field_value = 0
                
                # 인덱스 범위 초과 방지 (safety check)
                idx = int(field_value)
                if idx < 0: idx = 0
                if idx > 3: idx = 3

                if j < top: outside_map[0, idx] = 1
                if j > bottom: outside_map[1, idx] = 1
                if i < left: outside_map[2, idx] = 1
                if i > right: outside_map[3, idx] = 1

    features = wall_crates.flatten().tolist() + explosion_coins.flatten().tolist() \
             + bomb_opponents.flatten().tolist() + outside_map.ravel().tolist()

    return features
