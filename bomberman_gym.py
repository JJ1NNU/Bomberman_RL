import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment import BombeRLeWorld, WorldArgs # 기존 코드 활용
from agents import Agent # 기존 에이전트 클래스
from agent_code.my_irl_agent.features import state_to_features 


class BombermanEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action Space: 6개 (UP, RIGHT, DOWN, LEFT, WAIT, BOMB)
        self.action_space = spaces.Discrete(6)
        
        # Observation Space: CNN
        self.observation_space = spaces.Box(low=0, high=1, shape=(7, 17, 17), dtype=np.float32)
        
        # 게임 월드 초기화 (GUI 끄고 빠르게)
        args = WorldArgs(no_gui=True, fps=0, turn_based=False, update_interval=0, 
                         save_replay=False, replay=None, make_video=False, 
                         continue_without_training=True, log_dir="logs", save_stats=False, 
                         match_name="train", seed=None, silence_errors=True, scenario="classic")
        
        # 상대방은 룰베이스 에이전트로 채움
        agents = [("rule_based_agent", False)] * 3 
        # 주인공 에이전트 (우리가 학습시킬 대상)
        agents.insert(0, ("my_irl_agent", True)) 
        
        self.world = BombeRLeWorld(args, agents)
        self.my_agent = self.world.agents[0] # 주인공

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.new_round()
        
        # 첫 번째 상태 반환
        state = self.world.get_state_for_agent(self.my_agent)
        obs = state_to_features(state) # ★ 수정 필요
        return obs, {}

    def step(self, action_idx):
        actions_map = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        action_str = actions_map[action_idx]
        
        self.my_agent.backend.runner.fake_self.next_action = action_str

        self.world.do_step() 
        
        # 3. 결과 관측
        state = self.world.get_state_for_agent(self.my_agent)
        obs = state_to_features(state) # ★ 수정 필요
        reward = self.my_agent.score # 게임 내 점수를 쓸 수도 있지만 GAIL은 이걸 무시하고 자체 보상 생성함
        done = self.my_agent.dead or not self.world.running
        
        return obs, reward, done, False, {}
