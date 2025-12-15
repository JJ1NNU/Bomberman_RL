import functools
import os
import pickle
import random
import torch
import numpy as np
import math as m
import sys
from operator import itemgetter
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    global device
    if torch.backends.mps.is_available():
        device = torch.device("cpu")  # add mps if on mac
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    if self.train and not os.path.isfile("my-saved-model_ohne_batchnorm_neuer_reward5.pt"):
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()

    elif self.train and os.path.isfile("my-saved-model_ohne_batchnorm_neuer_reward5.pt"):
        print("Building on existing model.")

    else:
        print("Loading model from saved state.")
        with open("my-saved-model_ohne_batchnorm_neuer_reward5.pt", "rb") as file:
            self.model = pickle.load(file)

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

    # 좌표 및 행동 히스토리 관리 (매 라운드 초기화)
    if self.step == 1:
        self.coordinate_history.clear()
        self.action_history.clear()
        
    self.coordinate_history.append((self.x, self.y))

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기
    # -----------------------------------------------------------
    raw_action = _choose_action(self, game_state)

    # [★ 추가] 상자 옆에서 멍때림 방지 (Deadlock Breaking with Bomb)
    if game_state['self'][2] == True: # 폭탄 쿨타임 아님
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

        # print(f"Safety Shield Triggered! Intent: {raw_action} -> Adjusted: {final_action}")
        
    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)

    return final_action

def _choose_action(self, game_state: dict) -> str:
    """
    기존 act 함수의 핵심 로직을 분리한 함수
    """
    x1, x2, x3, x4 = state_to_features(game_state)

    if self.train:
        eps_start = 0.9
        eps_end = 0.1
        eps_decay = 1000
        round = 1
        sample = random.random()
        eps_threshold = 0.01 

        if sample > eps_threshold:
            round += 1
            with torch.no_grad():
                # self.policy_net이 정의되어 있지 않아 보이나 원본 로직 유지
                # 만약 policy_net이 없다면 model을 써야 함. 원본 코드의 문맥상 model이 쓰일 수도 있음.
                if hasattr(self, 'policy_net'):
                    action_done = torch.argmax(self.policy_net(x1, x2, x3, x4))
                else:
                    # fallback to model (loaded from pickle)
                    # pickle로 로드된 모델이 어떤 타입인지(torch model or numpy array)에 따라 다름
                    # 여기서는 원본 코드를 최대한 존중
                    if isinstance(self.model, torch.nn.Module):
                         action_done = torch.argmax(self.model(x1, x2, x3, x4))
                    else:
                         # 확률 분포인 경우 (초기 랜덤 설정 시)
                         action_done = np.argmax(self.model) 
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.2, .2, .2, .2, .1, .1])]],
                                        dtype=torch.long, device=device)
    else:
        # 평가 모드
        if isinstance(self.model, torch.nn.Module):
             action_done = torch.argmax(self.model(x1, x2, x3, x4))
        elif isinstance(self.model, np.ndarray):
             # setup에서 pickle.load로 불러온 것이 numpy array라면
             # 하지만 보통 torch 모델일 것임.
             # 원본 코드: torch.argmax(self.model(x1, x2, x3, x4)) -> torch 모델 가정
             action_done = torch.argmax(self.model(x1, x2, x3, x4))
        else:
             # 기타 타입일 경우 (예: 커스텀 클래스)
             action_done = torch.argmax(self.model(x1, x2, x3, x4))
    
    # 텐서에서 정수로 변환 필요
    if isinstance(action_done, torch.Tensor):
        action_index = action_done.item()
    else:
        action_index = action_done

    return ACTIONS[action_index]

def state_to_features(game_state: dict) -> np.array:
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

    outside_map = np.zeros((4, 4))  # coins, crates, bombs, other agents

    # --- map with walls (-1), free (0) and crates (1) ---------------------------------------------------------------
    min_coor = np.min(vision_coordinates, axis = 0)
    max_coor = np.max(vision_coordinates, axis = 0)
    wall_crates = np.zeros((reach * 2 + 1, reach * 2 + 1)) - 1  # outside of game also -1
    if 0 <= min(min_coor) and all(max_coor<field.shape):
        wall_crates[:, :] = field[min_coor[0]:max_coor[0]+1, min_coor[1]:max_coor[1]+1]
    else:
        wall_crates[abs(min(0,min_coor[0])):8-abs(min(0, field.shape[0]-2-max_coor[0])),abs(min(0,min_coor[1])):8-abs(min(0, field.shape[1]-2-max_coor[1]))] = field[max(0,min_coor[0]):min(max_coor[0]+1, field.shape[0]), max(0,min_coor[1]):min(max_coor[1]+1, field.shape[1])] 


    # --- map with explosion (-1) free (0) and coins (1) -------------------------------------------------------------
    explosion_coins = np.zeros((reach * 2 + 1, reach * 2 + 1))
    explosion_coord = np.transpose((explosion_map > 0).nonzero())

    existing_coords = explosion_coord[(explosion_coord[:, None] == vision_coordinates).all(-1).any(-1)]
    existing_coords[:, 0] = existing_coords[:,0]-my_pos[0]+reach
    existing_coords[:, 1] = existing_coords[:,1]-my_pos[1]+reach
    explosion_coins[existing_coords.T[0], existing_coords.T[1]] = -1

    if len(coins) > 0:
        coins = np.asarray(coins)
        existing_coins = coins[(coins[:, None] == vision_coordinates).all(-1).any(-1)]
        existing_coins[:, 0] = existing_coins[:,0]-my_pos[0]+reach
        existing_coins[:, 1] = existing_coins[:,1]-my_pos[1]+reach
        explosion_coins[existing_coins.T[0], existing_coins.T[1]] = 1

    # --- map with bomb range (-1), free (0) and opponents (1) --------------------------------------------------------
    bomb_opponents = np.zeros((reach * 2 + 1, reach * 2 + 1))
    vision_coordinates2 = vision_coordinates[:, np.newaxis, :]

    if len(others) > 0:
        others2 = np.array(list(map(itemgetter(3), others)))
        if len(others2) > 0:
            others2 = others2[np.newaxis, :, :]
            mask = np.all(vision_coordinates2 == others2, axis=2)
            mask = np.where(mask.any(axis=1))
            if len(mask) > 0:
                visable = vision_coordinates2[mask] - my_pos + reach
                bomb_opponents[tuple(visable.T)] = 1

    for bomb in bombs:
        # outside_map getting bombs outside vision field
        if bomb[0][1] < top:
            outside_map[2, 0] = 1
        if bomb[0][1] > bottom:
            outside_map[2, 1] = 1
        if bomb[0][0] < left:
            outside_map[2, 2] = 1
        if bomb[0][0] > right:
            outside_map[2, 3] = 1

        if any(sum(bomb[0] == i) == 2 for i in vision_coordinates):
            # coordinate of bomb in our vision matrix
            x_bomb = bomb[0][0] - my_pos[0] + reach
            y_bomb = bomb[0][1] - my_pos[1] + reach
            range_bomb = [1, 2, 3]

            bomb_opponents[x_bomb, y_bomb] = -1

            # compute the explosion range
            for j in range_bomb:
                if j + x_bomb > 6: break
                if wall_crates[x_bomb + j, y_bomb] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb + j, y_bomb] = -1

            for j in range_bomb:
                if j - x_bomb < 0: break
                if wall_crates[x_bomb - j, y_bomb] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb - j, y_bomb] = -1

            for j in range_bomb:
                if j + y_bomb > 6: break
                if wall_crates[x_bomb, y_bomb + j] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb, y_bomb + j] = -1

            for j in range_bomb:
                if j - y_bomb < 0: break
                if wall_crates[x_bomb, y_bomb - j] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb, y_bomb - j] = -1
        else: next

    # --- scaled down vision outside reach ---------------------------------------------------------------------------
    
    for i in range(0, field.shape[0]):
        for j in range(0, field.shape[1]):

            field_value = field[i, j]
            if field_value == 1:
                if j < top:
                    outside_map[1, 0] += 1
                if j > bottom:
                    outside_map[1, 1] += 1
                if i < left:
                    outside_map[1, 2] += 1
                if i > right:
                    outside_map[1, 3] += 1

    for coin in np.array(coins):
        if coin[1] < top:
            outside_map[0, 0] += 1
        if coin[1] > bottom:
            outside_map[0, 1] += 1
        if coin[0] < left:
            outside_map[0, 2] += 1
        if coin[0] > right:
            outside_map[0, 3] += 1

    others2 = np.array(list(map(itemgetter(3), others)))
    for enemy in others2:
        if enemy[1] < top:
            outside_map[3, 0] += 1
        if enemy[1] > bottom:
            outside_map[3, 1] += 1
        if enemy[0] < left:
            outside_map[3, 2] += 1
        if enemy[0] > right:
            outside_map[3, 3] += 1
    
    for i in range(0, explosion_map.shape[0]):
        for j in range(0, explosion_map.shape[1]):

            explosion_value = explosion_map[i, j]
            if explosion_value != 0:
                if j < top:
                    outside_map[2, 0] = 1
                if j > bottom:
                    outside_map[2, 1] = 1
                if i < left:
                    outside_map[2, 2] = 1
                if i > right:
                    outside_map[2, 3] = 1


    wall_crates = torch.tensor([wall_crates.tolist()], device=device)
    explosion_coins = torch.tensor([explosion_coins.tolist()], device=device)
    bomb_opponents = torch.tensor([bomb_opponents.tolist()], device=device)
    outside_map = torch.tensor([outside_map.ravel().tolist()], device=device)

    return wall_crates, explosion_coins, bomb_opponents, outside_map
