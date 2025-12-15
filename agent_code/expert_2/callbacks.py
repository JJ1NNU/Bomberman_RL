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

    self.coordinate_history = deque(maxlen=20)
    self.action_history = deque(maxlen=20) 
    
    self.step = 0
    
    if not hasattr(self, 'exploration_rate'):
        self.exploration_rate = 0.0

    if self.train:
        self.logger.info("Q-Learning algorithm.")
        self.name = "Table_1"
        self.number_of_states = len(self.valid_list)
        self.Q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 1.0
        self.exploration_rate_end = 0.05
        
        if not hasattr(self, 'n_rounds'):
            self.logger.warning("n_rounds not found in self. Setting default to 2000.")
            self.n_rounds = 1000
        
        self.exploration_decay_rate = set_decay_rate(self)
        self.exploration_rate = self.exploration_rate_initial

    else:
        self.logger.info("Loading from the latest Q_table")
        self.Q_table = load_latest_q_table(self, q_table_folder)
        if self.Q_table is None:
             self.logger.warning("No Q-Table found! Fallback to zeros.")
             self.Q_table = np.zeros(shape=(len(self.valid_list), len(ACTIONS)))
        self.exploration_rate = 0.0

def act(self, game_state: dict) -> str:
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

    # State Update
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

    # [★ 추가] 상자 옆에서 멍때림 방지 (Deadlock Breaking with Bomb)
    # 조건 1: 현재 폭탄 설치 가능 (game_state['self'][2] == True)
    # 조건 2: 최근 5턴간 폭탄을 놓지 않음 (이동만 반복 중)
    # 조건 3: 바로 옆(상하좌우)에 상자가 있음
    # 조건 4: ★ 폭탄을 놓아도 안전함 (Safety Shield가 허용함)

    if game_state['self'][2] == True:
        # 최근 기록 확인 (히스토리가 5개 미만이면 그냥 통과)
        recent_actions = list(self.action_history)[-5:]
        if len(recent_actions) >= 5 and all(a != 'BOMB' for a in recent_actions):
            
            # 상자 확인
            x, y = game_state['self'][3]
            field = game_state['field']
            neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            # 맵 밖 체크는 생략(가장자리엔 상자 없으므로), 1 = Crate
            crate_nearby = any(field[nx, ny] == 1 for nx, ny in neighbors if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1])
            
            if crate_nearby:
                # 안전 확인: get_filtered_actions를 미리 호출해봄
                try:
                    # action_history는 루프 감지용이지만, 여기선 폭탄 가능 여부만 볼 거라 그대로 넘김
                    temp_safe_actions = get_filtered_actions(game_state, self.action_history)
                    
                    if 'BOMB' in temp_safe_actions:
                        self.logger.info("Force BOMB placement: Stuck next to crate & Safe to bomb.")
                        raw_action = 'BOMB'
                        
                except Exception as e:
                    self.logger.error(f"Bomb Check Error: {e}")

    # -----------------------------------------------------------
    # 3. [Safety Shield] & [Loop Breaker]
    # -----------------------------------------------------------
    try:
        # get_filtered_actions 내부에서 Loop 감지 및 팀원 거리 유지를 수행함.
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

        self.logger.info(f"Safety Shield Triggered! Intent: {raw_action} -> Adjusted: {final_action}")
        
    # [중요] 결정된 행동을 히스토리에 저장
    self.action_history.append(final_action)

    return final_action

def _choose_action(self, state) -> str:
    if self.train and np.random.random() < self.exploration_rate:
        action = np.random.choice(ACTIONS)
        self.logger.info(f"act: Exploring: {action}")
        return action
    else:
        if self.Q_table is None:
             return np.random.choice(ACTIONS)
        action = ACTIONS[np.argmax(self.Q_table[state])]
        self.logger.info(f"act: Exploiting: {action}")
        return action

def state_to_features(self, game_state) -> int:
    features_dict = {}
    
    coin_direction = shortest_path_to_coin_or_crate(self, game_state)
    if coin_direction in ACTIONS_IDEAS:
        features_dict["Direction_coin/crate"] = coin_direction
    else:
        features_dict["Direction_coin/crate"] = 'UP' 

    bomb_safety_result = calculate_going_to_new_tiles(self, game_state)
    if bomb_safety_result in ["DOWN", "UP", "RIGHT", "LEFT", "SAFE"]:
        features_dict["Direction_bomb"] = bomb_safety_result
    elif bomb_safety_result == 'NO_OTHER_OPTION':
        random_choice = np.random.choice(ACTIONS_IDEAS)
        features_dict["Direction_bomb"] = random_choice
    else:
        features_dict["Direction_bomb"] = 'SAFE'

    features_dict["Place_Bomb"] = check_bomb_presence(self, game_state)
    features_dict["Crate_Radar"] = check_crate_presence(game_state)
    (features_dict["Up"], features_dict["Right"], features_dict["Down"], features_dict["Left"]) = compute_blockage(game_state)

    self.logger.info(f"Feature Dictionary: {features_dict}")

    for i, state in enumerate(self.valid_list):
        if state == features_dict:
            return i
            
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
