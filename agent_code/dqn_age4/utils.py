import torch
from .dqn_model import DQNAgent

def setup(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the state dimension matches the feature space (e.g., 350)
    self.agent = DQNAgent(state_dim=350, action_dim=6, device=device)
    self.replay_buffer = self.agent.replay_buffer 
    self.policy_net = self.agent.policy_net
    self.device = device
    self.logger.info("DQN agent setup completed.")
