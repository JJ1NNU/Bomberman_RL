import copy
import os
import pickle
import random
import sys
from collections import deque
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# [추가] 상위 경로 추가 (action_prune.py 접근용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

from .helpers import (look_for_targets, build_bomb_map, tile_value, coord_to_dir, find_targets2,
                      find_traps, best_explosion_score, explosion_score, passable, all_direction_distances,
                      guaranteed_passable_tiles, DIRECTIONS, bomb_explosion_map, is_safe
                      )
from .model import QNet

# if GPU is to be used
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 50
FORCE_BOMBS = False

def setup(self):
    """
    Set up your code. This is called once when loading each agent.
    """
    # [수정] 좌표 및 행동 히스토리 초기화
    self.coordinate_history = deque([], 20)
    self.action_history = deque([], 20) 
    self.shortest_way_coin = "None"
    self.shortest_way_crate = "None"
    self.shortest_way_safety = "None"
    self.steps = 0
    self.touching_crate = 0
    self.bomb_cooldown = 0

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = QNet(28, 1024, 1024, 6)
    else:
        self.logger.info("Loading model from saved state.")
        try:
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
        except Exception as e:
            self.logger.warning(f"Error loading model: {e}")
            self.model = QNet(28, 1024, 1024, 6) # Fallback

    self.logger.info(f"Using device: {device}")
    self.model.to(device)

# [수정] 기존 act 함수의 로직을 별도 함수로 분리 (Intent 파악용) -> 하지만 act 내에서 처리하는게 깔끔하므로 통합
# def _get_intent(self, game_state: dict) -> str: ... (삭제)

def act(self, game_state: dict) -> str:
    # -----------------------------------------------------------
    # 1. [Prepare] 피처 추출 및 상태 갱신
    # -----------------------------------------------------------
    self.features = state_to_features(self, game_state)
    self.x, self.y = game_state['self'][3]
    self.step = game_state['step']

    # 쿨타임 감소
    self.bomb_cooldown = max(0, self.bomb_cooldown - 1)

    # 좌표 및 행동 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
        self.action_history.clear()
        
    self.coordinate_history.append((self.x, self.y))
    if len(self.coordinate_history) > 20:
        self.coordinate_history.popleft() # pop(0) 대신 popleft() 사용 권장

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기
    # -----------------------------------------------------------
    raw_action = choose_action(self, game_state)

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
                        # self.logger.info("Force BOMB placement: Stuck next to crate & Safe to bomb.")
                        raw_action = 'BOMB'
                except Exception as e:
                    self.logger.error(f"Bomb Check Error: {e}")

    # -----------------------------------------------------------
    # 3. [Safety Shield] & [Loop Breaker]
    # -----------------------------------------------------------
    # [수정] 30스텝 조건 제거 -> 항상 쉴드 켜짐
    try:
        # [수정] action_prune에 action_history 전달
        safe_actions = get_filtered_actions(game_state, self.action_history)
    except Exception as e:
        self.logger.error(f"Safety Shield Error: {e}")
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

    # [로그] 최종 행동 결정 로그 (쉴드 발동 시에만)
    if final_action != raw_action:
        # self.logger.info(f"Safety Shield Triggered! Intent: {raw_action} -> Adjusted: {final_action}")
        pass
    else:
        self.logger.debug(f"Action execute: {final_action}")
    
    if game_state['bombs']:
        self.logger.info(f"Bombs: {game_state['bombs']}") 

    # -----------------------------------------------------------
    # 5. [Update] 쿨타임 업데이트 & 히스토리 저장
    # -----------------------------------------------------------
    if final_action == 'BOMB' and self.bomb_cooldown <= 0:
        self.bomb_cooldown = 7

    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)
 
    return final_action

def choose_action(self, game_state: dict) -> str:
    if FORCE_BOMBS and game_state['step'] % 20 == 19 and self.bomb_cooldown <= 0:
        self.logger.debug("Force dropped bomb.")
        return 'BOMB'

    # Explore random actions with probability epsilon
    rounds_done = game_state['round']
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * rounds_done / EPS_DECAY)

    if self.train and random.random() <= eps_threshold:
        self.logger.debug(f"Choosing action purely at random. Prob: {eps_threshold * 100:.2f} %")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = torch.tensor(self.features, dtype=torch.float).to(device)
    # 배치 차원 추가 (unsqueeze)
    if len(features.shape) == 1:
        features = features.unsqueeze(0)
        
    with torch.no_grad(): # 추론 시 그래디언트 계산 방지
        prediction = self.model(features)
    action = ACTIONS[torch.argmax(prediction).item()]

    self.logger.debug(f"Chose action {action}")

    return action

def state_to_features(self, game_state: dict) -> np.array:
    # ... (원본 state_to_features 함수 내용 그대로 유지) ...
    # 이 부분은 원본 파일의 state_to_features 함수 전체를 그대로 복사해 넣으세요.
    # 변경 사항 없음.
    if game_state is None:
        return None

    features = []

    # Gather information about the game state. Normalize to -1 <= x <= 1.
    # Arena 17 x 17 = 289
    field = game_state['field']

    explosions = game_state['explosion_map']
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)

    guaranteed_passable = guaranteed_passable_tiles(game_state)
    distance_map = guaranteed_passable_tiles(game_state, ignore_enemies=True)
    enemy_distances = guaranteed_passable_tiles(game_state, enemy_distances=True)

    empty_tiles = [(x, y) for x in cols for y in rows if (field[x, y] == 0)]
    bomb_map = build_bomb_map(game_state)
    safe_tiles = [tile for tile in empty_tiles if bomb_map[tile[0], tile[1]] == 100 and \
                  explosions[tile[0], tile[1]] == 0]

    # Score, Bomb_avail, Coordinates, Alone
    score_self = game_state['self'][1] / 100
    bomb_avail = int(game_state['self'][2])
    self_x, self_y = game_state['self'][3]
    self_x_normalized = self_x / 16
    self_y_normalized = self_y / 16

    features.append(bomb_avail)
    features.append(self_x_normalized)
    features.append(self_y_normalized)

    # In danger
    if bomb_map[self_x, self_y] == 100:
        in_danger = 0.0
    else:
        in_danger = 1.0

    features.append(in_danger)

    # Do not place suicidal bombs
    bomb_explosion = bomb_explosion_map(game_state, self_x, self_y)
    if np.all(np.logical_or(bomb_explosion == 1.0, guaranteed_passable < 0)):
        suicidal_bomb = 1.0
    else:
        suicidal_bomb = 0.0

    features.append(suicidal_bomb)

    # Distance to safety
    if in_danger == 1.0:
        safety_distances = find_targets2(guaranteed_passable, (self_x, self_y), safe_tiles)
        safety_distances = [d if d < 5 else -1 for d in safety_distances]
        if all(d == -1 for d in safety_distances):
            safety_distances = find_targets2(distance_map, (self_x, self_y), safe_tiles)
        # Normalize to -1 <= x <= 1
        safety_distances = [1 - (d / 32) if d >= 0 else -1 for d in safety_distances]
    else:
        safety_distances = [1.0] * 4

    # +4 features
    features.extend(safety_distances)

    # Avoid repetetive movement
    tile_freq = [0.0] * 4
    for i, direction in enumerate(DIRECTIONS):
        x2, y2 = self_x + direction[0], self_y + direction[1]
        if not passable(x2, y2, game_state):
            continue
        tile_freq[i] = 1 / (self.coordinate_history.count((x2, y2)) + 1)
    tile_freq_stay = 1 / (self.coordinate_history.count((self_x, self_y)) + 1)

    # +5 features
    features.extend(tile_freq)
    features.append(tile_freq_stay)

    # Distance to coins
    coins = game_state['coins']
    coin_distances = find_targets2(distance_map, (self_x, self_y), coins)
    # Normalize to -1 <= x <= 1
    coin_distances = [1 - (d / 32) if d >= 0 else -1 for d in coin_distances]

    # +4 features
    features.extend(coin_distances)

    # Avoid dangerous tiles
    safety = [0.0] * 4
    for i, direction in enumerate(DIRECTIONS):
        x2, y2 = self_x + direction[0], self_y + direction[1]
        if int(guaranteed_passable[x2, y2]) == 1 and is_safe(game_state, x2, y2):
            safety[i] = 1.0
    is_safe_stay = float(is_safe(game_state, self_x, self_y))

    # +5 features
    features.extend(safety)
    features.append(is_safe_stay)

    # TODO place good bombs
    # find best explosion direction
    max_steps = self.bomb_cooldown + 5
    explosion_score_up = best_explosion_score(game_state, bomb_map, (self_x, self_y), (0, -1), max_steps)
    explosion_score_right = best_explosion_score(game_state, bomb_map, (self_x, self_y), (1, 0), max_steps)
    explosion_score_down = best_explosion_score(game_state, bomb_map, (self_x, self_y), (0, 1), max_steps)
    explosion_score_left = best_explosion_score(game_state, bomb_map, (self_x, self_y), (-1, 0), max_steps)
    explosion_score_stay = explosion_score(game_state, bomb_map, self_x, self_y)

    explosion_scores = [explosion_score_up, explosion_score_right, explosion_score_down, explosion_score_left,
                        explosion_score_stay]

    best_explosion = np.argmax(explosion_scores[:4])
    # deepcopy 비용이 크지만 원본 로직 유지
    pot_game_state = copy.deepcopy(game_state) 
    pot_game_state['bombs'].append(((self_x, self_y), 5))
    if explosion_scores[best_explosion] == 0:
        best_explosion = -1
        self.shortest_way_crate = "None"
    elif explosion_scores[4] >= explosion_scores[best_explosion] and game_state['self'][2]:
        best_explosion = 4
        self.shortest_way_crate = "BOMB"
    else:
        self.shortest_way_crate = ACTIONS[best_explosion]

    explosion_scores = [float(i == best_explosion) for i in range(5)]
    if best_explosion == -1:
        crates = []
        for x in range(17):
            for y in range(17):
                if field[x, y] == 1:
                    crates.append((x, y))

        if len(crates) > 1:
            crate_dists = find_targets2(distance_map, (self_x, self_y), crates)
            closest_crate = np.argmin(crate_dists)
            explosion_scores = [float(i == closest_crate) for i in range(5)]

    (explosion_score_up, explosion_score_right,
     explosion_score_down, explosion_score_left, explosion_score_stay) = explosion_scores

    features.extend(explosion_scores)

    return features
