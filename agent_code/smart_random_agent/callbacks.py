import sys
import os
import numpy as np
import random
from collections import deque

# 상위 디렉토리(루트)를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from action_prune import get_filtered_actions

def setup(self):
    np.random.seed()
    self.prev_two_obs = deque([None, None], maxlen=2)


def act(self, game_state: dict):
    # 1. 안전한 행동 리스트 가져오기
    # [수정됨] 문자열 리스트 반환 (예: ['UP', 'BOMB', ...])
    safe_actions = get_filtered_actions(game_state, self.prev_two_obs)
    
    # 2. 관측 기록 업데이트
    self.prev_two_obs.append(game_state)
    
    # 3. 안전한 행동이 없으면 정지
    if not safe_actions:
        return 'WAIT'

    # 4. 필터링된 안전 행동 중 랜덤 선택 (폭탄 포함)
    # [수정됨] 문자열 그대로 반환
    return random.choice(safe_actions)
