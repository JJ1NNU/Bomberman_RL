import os
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet

from load_expert_data import load_and_convert_data
from bomberman_gym import BombermanEnv
import gym


# None 관측값 방어 래퍼
class NanGuardWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        if obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.array(obs, dtype=np.float32)

def make_env():
    env = BombermanEnv()
    env = NanGuardWrapper(env)
    env = Monitor(env)
    return env

# 2. Residual CNN
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.block(x)
        out = out + x  # skip connection
        return self.activation(out)


class BombermanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),                          # (32, 17, 17)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                          # (64, 9, 9)

            ResidualBlock(64),                  # (64, 9, 9)
            ResidualBlock(64),                  # (64, 9, 9)

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                          # (64, 5, 5)

            ResidualBlock(64),                  # (64, 5, 5)

            nn.Flatten(),
        )

        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# 환경 설정
print("🚀 Initializing Bomberman GAIL Training...")
env = DummyVecEnv([make_env])

# 전문가 데이터 로드
print("📊 Loading expert trajectories...")
trajectories = load_and_convert_data("dataset/expert_raw")
print(f"✅ Loaded {len(trajectories)} trajectories")

# 보상 네트워크
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=None,
)

# PPO Generator
print("🤖 Setting up PPO Generator...")
learner = PPO(
    env=env,
    policy="CnnPolicy",
    batch_size=64,
    n_steps=256,
    ent_coef=0.01,
    learning_rate=3e-4,
    n_epochs=8,
    verbose=1,
    policy_kwargs=dict(
        features_extractor_class=BombermanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    ),
)

# TensorBoard 로깅
logger = configure("./tb_logs/", ["stdout", "tensorboard", "csv"])
learner.set_logger(logger)

# GAIL Trainer
print("⚔️ Initializing GAIL Trainer...")
gail_trainer = GAIL(
    demonstrations=trajectories,
    demo_batch_size=512,
    gen_replay_buffer_capacity=4096,
    n_disc_updates_per_round=16,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True, 
)

# [수정] 안전한 진행 콜백
class SafeProgressCallback(BaseCallback):
    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0 and self.num_timesteps > 0:
            progress = (self.num_timesteps / self.total_steps) * 100
            print(f"🔄 Progress: {self.num_timesteps:,}/{self.total_steps:,} "
                  f"({progress:.1f}%) | FPS: {self.locals.get('fps', 0):.0f}")
        return True

# 체크포인트 + 진행 콜백
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./checkpoints/",
    name_prefix="gail_bomberman",
    verbose=1,
)

progress_callback = SafeProgressCallback(total_steps=20_000)

# 9. 학습 시작!
TOTAL_STEPS = 100_000
print(f"\n🎯 Starting GAIL Training for {TOTAL_STEPS:,} timesteps...")
print("📈 TensorBoard: tensorboard --logdir tb_logs/")
print("💾 Checkpoints: ./checkpoints/ (manual save recommended)")
print("=" * 80)

print("⚡ PPO 기본 로그로 실시간 진행 확인 중...")
gail_trainer.train(total_timesteps=TOTAL_STEPS)  # 콜백 제거!

print("💾 Emergency Save: 현재 상태 저장 중...")
learner.save("gail_bomberman_emergency")
print("✅ Emergency model saved: gail_bomberman_emergency.zip")
gail_trainer.reward_net.save("emergency_reward_net")
print("✅ Reward net also saved!")