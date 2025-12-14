from collections import namedtuple, deque
import pickle
from typing import List
import events as e
import random
import matplotlib.pyplot as plt
import torch
import numpy as np

from .callbacks import state_to_features
from .helpers import encode_action, plot, transform_action, transform_feature_vector
from .custom_events import *
from .model import QNet, DQN, DQN2

# This is only an example!
Memory = namedtuple('Memory',
                    ('state', 'action', 'reward', 'next_state', 'done'))

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu')

# Hyperparameters -- DO modify
MAX_MEMORY = 40_000
BATCH_SIZE = 128
LR = 0.001

plot_maxlen = 100
plt.style.use('Solarize_Light2')


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    """
    self.recent_scores = deque(maxlen=plot_maxlen)
    self.plot_scores = []
    self.plot_mean_scores = []
    self.total_score = 0
    self.epsilon = 0
    self.gamma = 0.95
    self.memory = deque(maxlen=MAX_MEMORY)
    self.trainer = DQN2(self.model, lr=LR, gamma=self.gamma, batch_size=BATCH_SIZE,
                        max_memory=MAX_MEMORY, device=device)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # -----------------------------------------------------------------------
    # 1. Feature Unpacking
    # -----------------------------------------------------------------------
    (bomb_avail, self_x_normalized, self_y_normalized, in_danger, suicidal_bomb,
     safety_distances_up, safety_distances_right, safety_distances_down, safety_distances_left,
     tile_freq_up, tile_freq_right, tile_freq_down, tile_freq_left,
     tile_freq_stay,
     coin_distances_up, coin_distances_right, coin_distances_down, coin_distances_left,
     safety_up, safety_right, safety_down, safety_left,
     safety_stay,
     explosion_score_up, explosion_score_right, explosion_score_down, explosion_score_left, explosion_score_stay,
     ) = tuple(self.features)

    safety_distances = [safety_distances_up, safety_distances_right, safety_distances_down, safety_distances_left]
    tile_freq = [tile_freq_up, tile_freq_right, tile_freq_down, tile_freq_left]
    coin_distances = [coin_distances_up, coin_distances_right, coin_distances_down, coin_distances_left]
    safety = [safety_up, safety_right, safety_down, safety_left]
    explosion_scores = [explosion_score_up, explosion_score_right, explosion_score_down, explosion_score_left]

    action_index = int(encode_action(self_action))

    # -----------------------------------------------------------------------
    # 2. Custom Events Calculation
    # -----------------------------------------------------------------------
    
    # (A) Agent did not wait
    if 'WAIT' not in events:
        events.append(NOT_WAITED)

    # (B) Loop Detection
    if self.coordinate_history.count((new_game_state['self'][3][0], new_game_state['self'][3][1])) >= 4:
        events.append(LOOP)
    else:
        events.append(NO_LOOP)

    # (C) Reward for moving towards coins
    if action_index < 4:
        if safety[action_index] == 1:
            events.append(SHORTEST_WAY_COIN)
        else:
            events.append(NOT_SHORTEST_WAY_COIN)

    # (D) Reward for staying safe (Moving towards safety)
    if in_danger:
        if action_index < 4:
            if safety[action_index] == 1:
                events.append(SHORTEST_WAY_SAFETY)
            else:
                events.append(NOT_SHORTEST_WAY_SAFETY)
        else:
            if safety_stay == 1:
                events.append(SHORTEST_WAY_SAFETY)
            else:
                events.append(NOT_SHORTEST_WAY_SAFETY)

    # (E) Good/Bad Bomb Logic (Simplified)
    # 폭탄을 놓았는데, 주변에 상자나 적이 있나?
    if 'BOMB_DROPPED' in events:
        my_x, my_y = old_game_state['self'][3]
        field = old_game_state['field']
        
        target_found = False
        
        # 상하좌우 3칸(폭발 범위) 탐색
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for dist in range(1, 4): # 1, 2, 3
                tx, ty = my_x + (dx * dist), my_y + (dy * dist)
                
                # 맵 밖이면 중단
                if not (0 <= tx < field.shape[0] and 0 <= ty < field.shape[1]):
                    break
                
                # 벽 만나면 폭발 끊김 -> 중단
                if field[tx, ty] == -1:
                    break
                
                # 상자 발견! -> Good Bomb
                if field[tx, ty] == 1:
                    target_found = True
                    break
                
                # 적 발견! -> Good Bomb
                # others: [(name, score, bombs_left, (x, y)), ...]
                for _, _, _, (ox, oy) in old_game_state['others']:
                    if (tx, ty) == (ox, oy):
                        target_found = True
                        break
            
            if target_found:
                break
        
        if target_found:
            events.append(GOOD_BOMB)
        else:
            events.append(BAD_BOMB)

    # (F) Step One Bomb check (Spawn kill prevention)
    if len(self.coordinate_history) == 1 and self_action == 'BOMB':
        events.append(STEP_ONE_BOMB)
    elif len(self.coordinate_history) == 1 and self_action != 'WAIT':
        events.append(NOT_STEP_ONE_BOMB)


    # -----------------------------------------------------------------------
    # 3. Calculate Reward & Train
    # -----------------------------------------------------------------------
    reward = reward_from_events(self, events, new_game_state['self'][1])

    # augment the dataset
    state_old_features = self.features
    state_new_features = state_to_features(self, new_game_state)

    old_features_t = transform_feature_vector(state_old_features)
    new_features_t = transform_feature_vector(state_new_features)
    act_t = transform_action(self_action)

    # encode_action
    action_enc = encode_action(self_action)
    act_enc_t = tuple([encode_action(act) for act in act_t])

    self.trainer.train(state_old_features, action_enc, reward, state_new_features, False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add custom events here
    score = last_game_state['self'][1]
    if score < 40:
        events.append(LOW_SCORING_GAME)
    if score > 45:
        events.append(HIGH_SCORING_GAME)
    if score >= 49:
        events.append(PERFECT_COIN_HEAVEN)

    # Reward for placement
    self_score = last_game_state['self'][1]
    opponent_scores = [opponent[1] for opponent in last_game_state['others']]
    higher_scores = [score for score in opponent_scores if score > self_score]
    self.placement = len(higher_scores) + 1

    match self.placement:
        case 1:
            events.append("FIRST")
        case 2:
            events.append("SECOND")
        case 3:
            events.append("THIRD")
        case 4:
            events.append("FOURTH")

    reward = reward_from_events(self, events, last_game_state['self'][1])

    # augment the dataset
    last_state_features = self.features
    last_features_t = transform_feature_vector(last_state_features)
    act_t = transform_action(last_action)

    # encode actions
    last_action_enc = encode_action(last_action)
    act_enc_t = tuple([encode_action(act) for act in act_t])

    # train the model
    self.trainer.train(last_state_features, last_action_enc, reward, last_state_features, True)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # plot results
    self.plot_scores.append(score)
    self.recent_scores.append(score)
    recent_mean_scores = sum(self.recent_scores) / len(self.recent_scores)
    self.plot_mean_scores.append(recent_mean_scores)
    plt.ion()
    plot(self.plot_scores, self.plot_mean_scores)


def reward_from_events(self, events: List[str], score) -> int:
    """
    통합된 보상 체계 (Unified Reward System)
    """
    game_rewards = {
        # 1. 생존 및 기본 행동
        e.KILLED_SELF: -100,         # [근거] 자폭은 학습을 망치는 최악의 행동. 절대 금지.
        e.GOT_KILLED: -50,           # [근거] 남한테 죽는 것도 큰 손해.
        e.INVALID_ACTION: -2,        # [근거] Safety Shield에 막히거나 벽에 박는 행동 감소.
        e.WAITED: -0.5,              # [근거] 불필요한 대기를 줄여 Loop 탈출 및 적극성 유도.
        e.SURVIVED_ROUND: +1,        # [근거] 오래 살아남는 것 자체에 대한 소소한 보상.
        
        # 2. 파밍 및 점수 획득
        e.COIN_COLLECTED: +5,        # [근거] 게임 승리의 핵심. 가장 큰 양수 보상.
        e.CRATE_DESTROYED: +3,       # [근거] 초반에 길을 열고 아이템을 얻기 위해 필수.
        e.COIN_FOUND: +2,            # [근거] 상자를 부숴서 코인을 발견하면 추가 이득.
        
        # 3. 폭탄 전략
        e.BOMB_DROPPED: -1,          # [근거] 무분별한 폭탄 설치 방지 (성공 시 +2+3 등으로 상쇄됨).
        e.BOMB_EXPLODED: 0,
        GOOD_BOMB: +2,               # [근거] 상자나 적을 노린 유효한 폭탄.
        BAD_BOMB: -2,                # [근거] 아무 의미 없는 허공 폭탄 처벌.
        STEP_ONE_BOMB: -5,           # [근거] 시작하자마자 자폭하는 '트롤링' 방지.
        NOT_STEP_ONE_BOMB: 0,
        
        # 4. 전투
        e.KILLED_OPPONENT: +15,      # [근거] 적 제거는 게임을 매우 유리하게 만듦.
        e.OPPONENT_ELIMINATED: +5,   # [근거] 적 탈락 보너스.
        TRAP: +10,
        MISSED_TRAP: -5,
        
        # 5. 내비게이션 (Navigation)
        # 길찾기를 돕는 보조 보상들
        SHORTEST_WAY_COIN: +1,
        NOT_SHORTEST_WAY_COIN: -0.5,
        SHORTEST_WAY_CRATE: +1,
        NOT_SHORTEST_WAY_CRATE: -0.5,
        
        # 6. 안전 및 패턴 관리
        SHORTEST_WAY_SAFETY: +3,     # [근거] 폭탄 설치 후 생존 본능 강화 (매우 중요).
        NOT_SHORTEST_WAY_SAFETY: -5, # [근거] 죽으러 가는 길은 강력하게 차단.
        LOOP: -2,                    # [근거] 제자리걸음 방지.
        NO_LOOP: 0,
        
        # 7. 게임 결과
        "FIRST": +20,
        "SECOND": +10,
        "THIRD": +5,
        "FOURTH": 0,
        LOW_SCORING_GAME: -5,
        HIGH_SCORING_GAME: +5,
        PERFECT_COIN_HEAVEN: +10
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
