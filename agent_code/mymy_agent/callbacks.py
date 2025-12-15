import numpy as np
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

from ..base_agent import main as base
from ..base_agent import features as base_features
from .features import get_features

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    base.setup(self, initialize, resume_training=True)
    if not hasattr(self, 'train'):
        self.train = False
    self.eval_mode = not self.train
    self.weights = np.array([0.1, 10, 1, 1, 1, 10, 1])
    
    # [수정] Loop Breaker용 히스토리
    self.coordinate_history = deque(maxlen=20) # 내부 로직용
    self.action_history = deque(maxlen=20)     # action_prune용
    self.step = 0

def initialize(self):
    self.features = [
        np.zeros((0, 1)), # Step
        np.zeros((0, 2)), # Closest coin
        np.zeros((0, 1)), # Nearby crates
        np.zeros((0, 2)), # Closest opponent
        np.zeros((0, 1)), # Nearby opponents
        np.zeros((0, 5)), # tile type
        np.zeros((0, 2))  # Direction of dense crates
    ]
    self.Q_table = np.zeros((0,6))

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

    # -----------------------------------------------------------
    # 3. [Safety Shield] & [Loop Breaker]
    # -----------------------------------------------------------
    # [수정] 30스텝 조건 제거 -> 항상 쉴드 켜짐
    try:
        # [수정] action_prune에 action_history 전달
        safe_actions = get_filtered_actions(game_state, self.action_history)
    except Exception as e:
        # logger가 없는 agent일 수 있으므로 print 사용
        print(f"Safety Shield Error: {e}") 
        safe_actions = ['WAIT'] # 비상시 정지

    final_action = raw_action
    
    # 쉴드에 의해 원래 의도가 차단되었는지 확인
    if raw_action not in safe_actions:
        
        # [수정] 원래 의도가 '폭탄'이 아니었다면, 대체 행동에서도 폭탄을 배제함.
        if raw_action != 'BOMB':
            # safe_actions에서 BOMB과 WAIT를 제외한 순수 이동만 후보로 선정
            candidate_actions = [a for a in safe_actions if a != 'BOMB' and a != 'WAIT']
        else:
            # 원래 폭탄을 놓으려다 막혔다면? (팀킬 방지 등)
            # -> 이동하거나 가만히 있거나 자유롭게 선택
            candidate_actions = [a for a in safe_actions if a != 'WAIT']

        # 후보가 있으면 그 중에서 랜덤 선택
        if candidate_actions:
            final_action = np.random.choice(candidate_actions)
            
        # 이동할 곳이 없으면 WAIT라도 해야지 (폭탄은 절대 안 됨)
        elif 'WAIT' in safe_actions:
            final_action = 'WAIT'
            
        # WAIT도 없고 폭탄만 남았다면? (드문 경우)
        elif safe_actions: 
            final_action = np.random.choice(safe_actions)
        else:
            final_action = 'WAIT'

        # logger가 있으면 사용, 없으면 무시
        if hasattr(self, 'logger'):
            self.logger.info(f"Safety Shield Triggered! Intent: {raw_action} -> Adjusted: {final_action}")
            
    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)

    return final_action

def _choose_action(self, game_state: dict) -> str:
    """
    기존 act 함수의 핵심 로직
    """
    game_state_transformed, transformations = base.get_standarized_state_dict(game_state)
    
    field = game_state_transformed['field'].copy()
    field = base.add_coords_to_matrix(field, np.array([other[3] for other in game_state_transformed['others']]), value=1)
    field = base.add_coords_to_matrix(field, np.array([bomb[0] for bomb in game_state_transformed['bombs']]), value=1)
    
    valid_moves = base.get_valid_moves(
        field,
        game_state_transformed['self'][3],
        bomb_possible=game_state_transformed['self'][2],
        allow_wait=True
    )
    
    safe_moves = base.get_safe_moves(game_state_transformed, valid_moves)
    
    if self.train and not self.eval_mode and np.random.random() < base.epsion_greedy(self, initial_eps=0.9, final_eps=0.1, decay=0.005):
        selected_move = base.select_move(np.random.random(6), safe_moves)
    else:
        features = self.new_features if (self.train and not self.eval_mode and game_state_transformed['step'] != 1) else get_features(game_state_transformed)
        
        idx, is_known = base_features.get_state_id(self, features)
        
        if is_known:
            not_visited = self.Q_table[idx] == 0
            need_regression = (not_visited * safe_moves).any()
            
            if need_regression:
                regression_values = base.Q_regression(self, features, is_known)
                regression_values = self.Q_table[idx] * (1 - not_visited) + regression_values * not_visited
                selected_move = base.select_move(regression_values, safe_moves)
            else:
                selected_move = base.select_move(self.Q_table[idx], safe_moves)
        else:
            regression_values = base.Q_regression(self, features, is_known)
            selected_move = base.select_move(regression_values, safe_moves)

    return base.revert_move_transformations(selected_move, transformations)
