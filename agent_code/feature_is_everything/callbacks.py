import pickle
import random
import torch
import os
import sys
import numpy as np
from collections import deque

# [추가] 상위 경로 추가 (action_prune.py 접근용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

from .features import Feature

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def encode_feature(features):
    if features is None or len(features) == 0:
        return [0] * 34
    
    mapping_0_4 = {'block': 0, 'free': 1, 'dead': 2, 'coin': 3, 'enemy': 4, 'target': 5}
    mapping_5 = {'True': 0, 'False': 1, 'target': 2, 'KILL!': 3}
    
    onehot_vector = []
    
    for i in range(5):
        onehot = [0] * 6
        value = features[i]
        if value in mapping_0_4:
            idx = mapping_0_4[value]
            onehot[idx] = 1
        onehot_vector.extend(onehot)
        
    onehot = [0] * 4
    value = features[5]
    if value in mapping_5:
        idx = mapping_5[value]
        onehot[idx] = 1
    onehot_vector.extend(onehot)
    
    return onehot_vector

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.feature = Feature()

    # [추가] Loop Breaker 및 좌표 추적을 위한 초기화
    self.coordinate_history = deque(maxlen=20)
    self.step = 0
    
    # [추가] Act에서 사용하는 변수 안전 초기화 (원본 코드에 누락된 경우 대비)
    if not hasattr(self, 'epsilon'):
        self.epsilon = 0.1
    if not hasattr(self, 'steps_done'):
        self.steps_done = 0

    if not self.train:
        # 모델 로드 실패 대비
        if os.path.isfile("my-saved-model.pt"):
            with open("my-saved-model.pt", "rb") as file:
                self.policy_net = pickle.load(file).to(self.device)
            self.logger.info("Loading model from saved state.")
        else:
            self.logger.warning("Model file not found! Training mode or broken setup?")
            # Fallback: 모델이 없으면 act에서 에러나므로 더미 모델이라도 있어야 함
            # 하지만 보통은 학습된 파일이 있다고 가정.
            self.policy_net = None 

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
        
        # [원본 로직 유지] If train with itself (특정 조건에서 재 setup)
        if game_state["round"] % 500 == 0:
            setup(self)

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
            self.logger.error(f"Safety Shield Error: {e}")
            safe_actions = [raw_action]
            
        # -----------------------------------------------------------
        # 4. [Loop Breaker] (30스텝 이후)
        # -----------------------------------------------------------
        is_looping = False
        is_safe_now = 'WAIT' in safe_actions
        
        if is_safe_now and len(self.coordinate_history) >= 10:
            history_list = list(self.coordinate_history)
            recent_locs = set(history_list[-10:])
            if len(recent_locs) <= 2:
                is_looping = True
        
        final_action = raw_action
        
        # 루프 탈출 또는 위험 행동 차단
        if is_looping or (raw_action not in safe_actions):
            if is_looping:
                pass 
            
            if safe_actions:
                safe_moves = [a for a in safe_actions if a != 'WAIT']
                if safe_moves:
                    final_action = np.random.choice(safe_moves)
                else:
                    final_action = np.random.choice(safe_actions)
            else:
                final_action = 'WAIT'

    if self.train:
        self.steps_done += 1

    return final_action

def _choose_action(self, game_state: dict) -> str:
    """
    기존 act 함수의 핵심 로직 분리
    """
    features = self.feature(game_state)
    features_tensor = torch.tensor(encode_feature(features), dtype=torch.float32).unsqueeze(0).to(self.device)

    # 모델이 없는 경우 (파일 로드 실패 등) 랜덤
    if not hasattr(self, 'policy_net') or self.policy_net is None:
        return np.random.choice(ACTIONS)

    if self.train:
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(features_tensor)
                action_index = q_values.max(1)[1].item()
        else:
            action_index = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            q_values = self.policy_net(features_tensor)
            self.logger.debug(f"q_values: {q_values}")
            action_index = q_values.max(1)[1].item()
            
    return ACTIONS[action_index]
