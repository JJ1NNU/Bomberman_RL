import os
import sys
import numpy as np
import math
import itertools
import re
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

from .features import check_bomb_presence, check_crate_presence, compute_blockage, calculate_going_to_new_tiles, \
    shortest_path_to_coin_or_crate

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_IDEAS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def setup(self):
    q_table_folder = "Q_tables/"
    self.valid_list = Valid_States()
    self.new_state = None
    self.old_state = None
    self.old_distance = 0
    self.new_distance = 0

    # [추가] Loop Breaker 및 좌표 추적을 위한 초기화
    self.coordinate_history = deque(maxlen=20)
    self.step = 0
    # [추가] exploration_rate 안전 초기화
    if not hasattr(self, 'exploration_rate'):
        self.exploration_rate = 0.0

    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.name = "Table_1"
        self.number_of_states = len(self.valid_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS))) # number_of_states * 6
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05
        
        if not hasattr(self, 'n_rounds'):
            self.logger.warning("n_rounds not found in self. Setting default to 2000.")
            self.n_rounds = 1000
        
        self.exploration_decay_rate = set_decay_rate(self)
        
        # [수정] 훈련 시작 시 exploration rate 초기화
        self.exploration_rate = self.exploration_rate_initial

    else:
        self.logger.info("Loading from the latest Q_table")
        self.Q_table = load_latest_q_table(self, q_table_folder)
        # 만약 로드 실패시 안전장치
        if self.Q_table is None:
             self.logger.warning("No Q-Table found! Fallback to zeros.")
             self.Q_table = np.zeros(shape=(len(self.valid_list), len(ACTIONS)))
        self.exploration_rate = 0.0 # 테스트 모드

def act(self, game_state: dict) -> str:
    
    # -----------------------------------------------------------
    # 1. [Prepare] 상태 갱신
    # -----------------------------------------------------------
    self.step = game_state['step']
    self.x, self.y = game_state['self'][3]

    # 좌표 히스토리 관리
    if self.step == 1:
        self.coordinate_history.clear()
    self.coordinate_history.append((self.x, self.y))

    # State Update (Original Logic)
    if self.new_state is None:
        self.old_state = state_to_features(self, game_state)
    else:
        self.old_state = self.new_state
    
    state = self.old_state
    self.logger.info(f"act: State: {state}")

    # -----------------------------------------------------------
    # 2. [Intent] 모델에게 원래 의도 물어보기
    # -----------------------------------------------------------
    raw_action = _choose_action(self, state)

    # -----------------------------------------------------------
    # 3. [Early Game Check] & [Safety Shield]
    # -----------------------------------------------------------
    if self.step < 30:
        self.logger.info(f"Early Game (Step {self.step}): Safety Shield is OFF.")
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
                self.logger.info("Loop detected in SAFE state. Forcing random move.")
            
            if safe_actions:
                safe_moves = [a for a in safe_actions if a != 'WAIT']
                if safe_moves:
                    final_action = np.random.choice(safe_moves)
                else:
                    final_action = np.random.choice(safe_actions)
            else:
                final_action = 'WAIT'

    # [로그] 최종 행동 결정 로그
    if final_action != raw_action:
        self.logger.info(f"Safety Shield Triggered! Intent: {raw_action} -> Adjusted: {final_action}")
    else:
        self.logger.debug(f"Action execute: {final_action}")

    return final_action

def _choose_action(self, state) -> str:
    """
    기존 act 함수의 선택 로직 분리
    """
    # Exploration
    if self.train and np.random.random() < self.exploration_rate:
        action = np.random.choice(ACTIONS)
        self.logger.info(f"act: Exploring: {action}")
        return action
    
    # Exploitation
    else:
        # Q-Table이 None인 경우 방어
        if self.Q_table is None:
             return np.random.choice(ACTIONS)
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"act: Exploiting: {action}")
        return action

def state_to_features(self, game_state) -> int:
    features_dict = {}
    
    # Feature 1: Direction_coin/crate
    coin_direction = shortest_path_to_coin_or_crate(self, game_state)
    if coin_direction in ACTIONS_IDEAS:
        features_dict["Direction_coin/crate"] = coin_direction
    else:
        self.logger.info(f"!!! state_to_features: shortest_path_to_coin_or_crate: Invalid direction: {coin_direction}")
        # Default fallback to prevent crash if logic fails
        features_dict["Direction_coin/crate"] = 'UP' 

    # Feature 2: Direction_bomb
    bomb_safety_result = calculate_going_to_new_tiles(self, game_state)
    if bomb_safety_result in ["DOWN", "UP", "RIGHT", "LEFT", "SAFE"]:
        features_dict["Direction_bomb"] = bomb_safety_result
    elif bomb_safety_result == 'NO_OTHER_OPTION':
        random_choice = np.random.choice(ACTIONS_IDEAS)
        self.logger.info(f"calculate_going_to_new_tiles: No shortest path {random_choice}")
        features_dict["Direction_bomb"] = random_choice
    else:
        self.logger.info(f"!!! state_to_features: calculate_going_to_new_tiles: Invalid direction: {bomb_safety_result}")
        features_dict["Direction_bomb"] = 'SAFE'

    # Feature 3: Place_Bomb
    features_dict["Place_Bomb"] = check_bomb_presence(self, game_state)

    # Feature 4: Crate_Radar
    features_dict["Crate_Radar"] = check_crate_presence(game_state)

    # Feature 5: All directions actions
    (features_dict["Up"], features_dict["Right"], features_dict["Down"], features_dict["Left"]) = compute_blockage(game_state)

    self.logger.info(f"Feature Dictionary: {features_dict}")

    for i, state in enumerate(self.valid_list):
        if state == features_dict:
            return i
            
    # If state not found (should not happen if Valid_States covers all)
    self.logger.error("State dict created by state_to_features was not found in self.valid_list")
    return 0 

def load_latest_q_table(self, q_table_directory):
    try:
        if not os.path.exists(q_table_directory):
             self.logger.info("load_latest_q_table: Q-table directory not found.")
             return None
             
        files = os.listdir(q_table_directory)
        q_table_files = [file for file in files if file.startswith("Q_table-")]
        
        if not q_table_files:
            self.logger.info("load_latest_q_table: No Q-table files found in the directory.")
            return None
        
        # Extracting the numbers from the filenames.
        numbers = [int(re.search(r'\d+', file).group()) for file in q_table_files if re.search(r'\d+', file)]
        if not numbers:
             return None
             
        latest_q_table_number = max(numbers)
        latest_q_table_file = f"Q_table-Table_{latest_q_table_number}.npy"
        latest_q_table_path = os.path.join(q_table_directory, latest_q_table_file)
        
        q_table = np.load(latest_q_table_path)
        self.logger.info(f"load_latest_q_table: Q-table file loaded:{latest_q_table_path}")
        return q_table
    except Exception as e:
        self.logger.error(f"load_latest_q_table: Error loading table: {e}")
        return None

def set_decay_rate(self) -> float:
    decay_rate = -math.log((self.exploration_rate_end + 0.005) / self.exploration_rate_initial) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate

def Valid_States():
    feature_list = []
    # (원본의 tuple 구조 유지)
    valid_states = list(itertools.product(('UP', 'RIGHT', 'DOWN', 'LEFT'), ('UP', 'RIGHT', 'DOWN', 'LEFT', 'SAFE'),
                                          ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'), ('MOVE', 'BLOCK'),
                                          ('YES', 'NO'), ('LOW', 'MID', 'HIGH')))
    for states in valid_states:
        features = {
            "Direction_coin/crate": states[0],
            "Direction_bomb": states[1],
            "Up": states[2],
            "Right": states[3],
            "Down": states[4],
            "Left": states[5],
            "Place_Bomb": states[6],
            "Crate_Radar": states[7]
        }
        feature_list.append(features)
    return feature_list
