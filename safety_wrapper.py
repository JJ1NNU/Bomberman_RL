import random
from action_prune import get_filtered_actions # 제공된 파일 import
import settings as s

class SafetyShieldWrapper:
    def __init__(self, agent_logic):
        self.agent = agent_logic # 기존 에이전트 객체

    def act(self, observation, action_space):
        # 1. 기존 에이전트의 의도(Intent) 파악
        raw_action = self.agent.act(observation, action_space)
        
        # 2. Skynet의 Action Pruning 적용
        # get_filtered_actions는 안전한 행동의 인덱스 리스트를 반환함
        safe_actions = get_filtered_actions(observation)
        
        # 3. 마스킹 로직
        if raw_action in safe_actions:
            return raw_action
        else:
            # 의도한 행동이 위험하다면, 안전한 행동 중 하나 선택 (혹은 STOP)
            if not safe_actions:
                return s.Actions.Stop # 살 방법이 없으면 정지
            return random.choice(safe_actions)
