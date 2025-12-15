import os
import sys
import numpy as np

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # [추가]

# 1. 단일 환경 생성 및 Monitor 래핑 (종료 시 정보 보존)
def make_env():
    env = BombermanEnv()
    env = Monitor(env)  # 학습 로그 및 info 보정
    return env

class BombermanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 차원 계산
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# 1. 파일 존재 여부 체크
required_files = [
    "bomberman_gym.py",
    "load_expert_data.py",
    "train_gail.py",
    "agent_code/my_irl_agent/features.py",
    "agent_code/my_irl_agent/callbacks.py"
]

print("=== [1] Checking File Existence ===")
missing_files = []
for f in required_files:
    if os.path.exists(f):
        print(f"[OK] Found {f}")
    else:
        print(f"[FAIL] Missing {f}")
        missing_files.append(f)

if missing_files:
    print("!!! Missing files detected. Please create or move them first.")
    sys.exit(1)

# 2. 모듈 임포트 테스트
print("\n=== [2] Checking Imports ===")
try:
    from bomberman_gym import BombermanEnv
    from load_expert_data import load_and_convert_data
    from agent_code.my_irl_agent.features import state_to_features
    print("[OK] All modules imported successfully.")
except ImportError as e:
    print(f"[FAIL] Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Unexpected Error during import: {e}")
    sys.exit(1)

# 3. 데이터 로딩 테스트
print("\n=== [3] Checking Data Loading ===")
data_dir = "dataset/expert_raw"
if not os.path.exists(data_dir) or not os.listdir(data_dir):
    print(f"[WARN] No expert data found in {data_dir}. Creating dummy data for test.")
    # (테스트용 더미 파일 생성이 필요하다면 추가, 일단 경고만)
else:
    try:
        # 1개 파일만 로드 시도 (시간 절약)
        # load_and_convert_data 함수가 인자로 파일 개수 제한을 받지 않는다면 
        # 일단 호출해보고 너무 오래 걸리면 강제 종료해야 함.
        # 여기서는 정상 호출 되는지만 확인
        trajectories = load_and_convert_data(data_dir)
        print(f"[OK] Loaded {len(trajectories)} trajectories.")
        if len(trajectories) > 0:
            print(f"    - Obs shape: {trajectories[0].obs.shape}")
            print(f"    - Acts shape: {trajectories[0].acts.shape}")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        # sys.exit(1) # 데이터가 없어도 환경 테스트는 진행 가능하므로 주석 처리

# 4. Gym 환경 테스트 (가장 중요)
print("\n=== [4] Checking Gym Environment ===")
try:
    env = BombermanEnv()
    print("[OK] Environment initialized.")
    
    # Reset 테스트
    obs, info = env.reset()
    print(f"[OK] Reset successful. Obs shape: {obs.shape}")
    
    if obs.shape != (7, 17, 17):
        print(f"[FAIL] Observation shape mismatch! Expected (7, 17, 17), got {obs.shape}")
    
    # Step 테스트 (행동 주입 확인)
    action = env.action_space.sample() # 0~5 랜덤
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"[OK] Step successful. Reward: {reward}, Done: {done}")
    
    # features.py와 callbacks.py가 잘 연결되었는지 확인
    # (에러 없이 여기까지 왔다면 연결 성공)

except Exception as e:
    print(f"[FAIL] Environment Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. GAIL 학습 루프 테스트 (Dummy Training)
print("\n=== [5] Checking GAIL Training Loop (Dry Run) ===")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    
    # 더미 데이터가 없으면 테스트 불가하므로 생성
    if 'trajectories' not in locals() or len(trajectories) == 0:
        print("[INFO] Creating dummy trajectories for training test...")
        from imitation.data.types import Trajectory
        dummy_obs = np.random.rand(10, 7, 17, 17)
        dummy_acts = np.random.randint(0, 6, size=(10,))
        trajectories = [Trajectory(obs=np.vstack([dummy_obs, dummy_obs[-1][None, ...]]), acts=dummy_acts, infos=None, terminal=True)]

    venv = DummyVecEnv([make_env])
    
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=None,
    )
    
    learner = PPO(
        env=venv,
        policy="CnnPolicy",
        batch_size=4, # 테스트용 소형 배치
        n_steps=8, # 테스트용 소형 itr
        n_epochs=1,
        learning_rate=0.0003,
         policy_kwargs=dict(
        features_extractor_class=BombermanCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False
    ),
    )
    
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=4, # 테스트용
        gen_replay_buffer_capacity=10,
        n_disc_updates_per_round=1,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )
    
    print("Starting short training (10 steps)...")
    gail_trainer.train(total_timesteps=10) # 딱 10스텝만 학습
    print("[OK] GAIL training loop completed successfully.")

except Exception as e:
    print(f"[FAIL] Training Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ ALL SYSTEMS GO! Everything seems correct.")
