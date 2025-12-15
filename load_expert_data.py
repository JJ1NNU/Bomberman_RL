import pickle
import glob
import numpy as np
import torch
from imitation.data.types import Trajectory
from agent_code.my_irl_agent.features import state_to_features

def load_and_convert_data(data_dir="dataset/expert_raw"):
    files = glob.glob(f"{data_dir}/*.pkl")
    all_trajectories = []
    
    print(f"Loading {len(files)} expert files...")
    
    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                raw_data = pickle.load(f) # [(state, action), ...]
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")
            continue
            
        obs_list = []
        acts_list = []
        
        for state, action in raw_data:
            # 1. State -> Feature Vector 변환
            # state_to_features(self, game_state) 시그니처 대응
            # self가 필요 없다면 None을 전달, 필요하다면 더미 객체 생성 필요할 수도 있음.
            # features.py 구현에 따라 다름. 보통 self를 안 쓰거나 로거용으로만 씀.
            try:
                feature_vector = state_to_features(None, state) 
            except TypeError:
                # 만약 state_to_features(game_state) 형태라면:
                feature_vector = state_to_features(state)
            
            # 2. Action -> Index 변환
            actions_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
            if action not in actions_map:
                continue # 알 수 없는 액션이면 스킵
                
            action_idx = actions_map[action]
            
            obs_list.append(feature_vector)
            acts_list.append(action_idx)
            
        # Trajectory 객체 생성
        if len(obs_list) > 0:
            # [중요] dtype=np.float32 명시
            obs_array = np.array(obs_list, dtype=np.float32) 
            
            # [중요] 마지막 관측값 차원 맞춰서 붙이기
            # obs_array[-1]은 (C, H, W) -> [None, ...] 하면 (1, C, H, W)
            last_obs = obs_array[-1][None, ...] 
            
            full_obs = np.vstack([obs_array, last_obs]) # (N+1, C, H, W)
            
            traj = Trajectory(
                obs=full_obs,
                acts=np.array(acts_list, dtype=np.int64), # 액션은 int64 추천
                infos=None,
                terminal=True
            )
            all_trajectories.append(traj)
            
    print(f"Total {len(all_trajectories)} trajectories loaded.")
    return all_trajectories
