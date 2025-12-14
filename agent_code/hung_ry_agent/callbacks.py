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
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.eval_mode = True
    base.setup(self, initialize, resume_training=False)
    self.weights = np.array([0.1, 10, 1, 1, 1, 10, 1])
    self.coordinate_history = deque(maxlen=20)
    self.step = 0

def initialize(self):
    self.features = [
        np.zeros((0, 1)), # Step
        np.zeros((0, 2)), # Closest coin
        np.zeros((0, 1)), # Nearby crates
        np.zeros((0, 2)), # Closest opponent
        np.zeros((0, 1)), # Nearby opponents
        #np.zeros((0, 4, 2)), # Bombs
        np.zeros((0, 5)), # tile type
        np.zeros((0, 2)) # Direction of dense crates
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

    # 좌표 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
    self.coordinate_history.append((self.x, self.y))

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기 (복잡한 로직 위임)
    # -----------------------------------------------------------
    # 주의: _choose_action 내부에서 원본 act의 모든 로직(변환 포함)을 수행하고
    # 최종적으로 "변환이 복구된" 실제 행동(Action String)을 반환받습니다.
    raw_action = _choose_action(self, game_state)

    # -----------------------------------------------------------
    # 3. [Early Game Check] & [Safety Shield]
    # -----------------------------------------------------------
    if self.step < 30:
        # 초반 30스텝은 Safety Shield 끄기 (자폭 방지 등은 _choose_action 내부 로직에 맡김)
        final_action = raw_action
    else:
        # 30스텝 이후 Safety Shield 작동
        try:
            # get_filtered_actions는 원본 game_state(변환 전)를 사용해야 함
            safe_actions = get_filtered_actions(game_state)
        except Exception as e:
            # 에러 발생 시 원본 행동 유지
            print(f"Safety Shield Error: {e}") # logger가 없는 경우 대비
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
                pass # 루프 감지됨
            
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
    기존 act 함수의 핵심 로직(State Transformation, Q-Regression, Revert 등)을 그대로 분리
    """
    # [원본 로직 시작]
    # game_state 복사를 방지하기 위해 원본 로직이 dict를 수정하는지 확인해야 하지만,
    # 여기서는 안전하게 로직을 수행합니다.
    
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
    
    play_random = False
    
    if play_random:
        selected_move = base.select_move(np.random.random(6), safe_moves)
    else:
        if self.train and not self.eval_mode and np.random.random() < base.epsion_greedy(self, initial_eps=0.9, final_eps=0.1, decay=0.005):
            selected_move = base.select_move(np.random.random(6), safe_moves)
        else:
            # features 추출 시 원본 game_state가 아닌 변환된 것이 필요한지, 원본이 필요한지
            # 원본 코드: get_features(game_state) -> 여기서 game_state는 act함수의 인자(원본)였으나,
            # 첫 줄에서 game_state, transformations = base... 로 덮어씌워짐.
            # 따라서 변환된 game_state_transformed를 써야 함. (변수명 주의)
            
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

    # 최종적으로 변환된 좌표계에서의 행동(selected_move)을 원래대로 복구하여 반환
    return base.revert_move_transformations(selected_move, transformations)