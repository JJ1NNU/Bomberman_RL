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
    # 2. [Intent] 모델에게 원래 의도 물어보기 (기존 로직)
    # -----------------------------------------------------------
    raw_action = _choose_action(self, game_state)

    # -----------------------------------------------------------
    # 3. [Early Game Check] & [Safety Shield]
    # -----------------------------------------------------------
    # 경기 초반 (30스텝 이전)에는 안전장치 비활성화
    if self.step < 30:
        # [주의] 초반이라도 자폭/팀킬 방지를 위해 최소한의 체크가 필요할 수 있으나,
        # 일단 기준에 맞춰 Shield OFF
        final_action = raw_action
    else:
        # 30스텝 이후 Safety Shield 작동
        try:
            safe_actions = get_filtered_actions(game_state)
        except Exception as e:
            # 에러 발생 시 로그 찍고 원본 행동 유지 (여기선 logger가 없으므로 print)
            print(f"Safety Shield Error: {e}")
            safe_actions = [raw_action]
            
        # -----------------------------------------------------------
        # 4. [Loop Breaker] (30스텝 이후)
        # -----------------------------------------------------------
        is_looping = False
        is_safe_now = 'WAIT' in safe_actions # WAIT가 안전하다는 건 급박하지 않다는 뜻
        
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
                pass # 루프 감지됨 (로그 없음)
            
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
                action_done = torch.argmax(self.policy_net(x1, x2, x3, x4))
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.2, .2, .2, .2, .1, .1])]],
                                       dtype=torch.long, device=device)
    else:
        # 평가 모드
        action_done = torch.argmax(self.model(x1, x2, x3, x4))
    
    return ACTIONS[action_done]

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
