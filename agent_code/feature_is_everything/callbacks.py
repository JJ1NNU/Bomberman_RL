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

    # [추가] Loop Breaker 및 좌표/행동 추적을 위한 초기화
    self.coordinate_history = deque(maxlen=20)
    self.action_history = deque(maxlen=20) 
    self.step = 0
    
    # [추가] Act에서 사용하는 변수 안전 초기화
    if not hasattr(self, 'epsilon'):
        self.epsilon = 0.1
    if not hasattr(self, 'steps_done'):
        self.steps_done = 0

    if not self.train:
        # 모델 로드 실패 대비
        if os.path.isfile("my-saved-model.pt"):
            try:
                with open("my-saved-model.pt", "rb") as file:
                    self.policy_net = pickle.load(file).to(self.device)
                self.logger.info("Loading model from saved state.")
            except Exception as e:
                self.logger.warning(f"Error loading model: {e}")
                self.policy_net = None
        else:
            self.logger.warning("Model file not found! Training mode or broken setup?")
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

    # 좌표 및 행동 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
        self.action_history.clear()
        
        # [원본 로직 유지] If train with itself (특정 조건에서 재 setup)
        if game_state["round"] % 500 == 0:
            setup(self)

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

    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)

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
