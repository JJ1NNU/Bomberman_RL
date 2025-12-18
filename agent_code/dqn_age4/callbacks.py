import numpy as np
import torch
from .features import state_to_features
from .utils import setup
from .dqn_model import DQNAgent

def setup(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the state dimension matches the feature space (e.g., 350)
    self.agent = DQNAgent(state_dim=350, action_dim=6, device=device)
    self.replay_buffer = self.agent.replay_buffer 
    self.policy_net = self.agent.policy_net
    self.device = device
    self.logger.info("DQN agent setup completed.")
    if not self.train:  # If not in training mode, load the trained model
        self.agent.load_model("latest_model.pth")
        self.logger.info("Loaded trained model for play mode.")
    self.logger.info("DQN Agent setup complete.")

# Example place in your code where you might prepare the state
def prepare_state(state):
    # Convert state to a numpy array if it's a list of arrays
    if isinstance(state, list):
        state = np.array(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert numpy array to tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    return state_tensor

# def act(self, game_state):
#     # Convert the game state to features
#     state = state_to_features(game_state)

#     # Ensure device is set
#     if not hasattr(self, 'device'):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Convert the state to a tensor and move it to the appropriate device
#     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

#     # Check if the policy_net is set properly
#     if not hasattr(self, 'policy_net'):
#         raise AttributeError("policy_net is not initialized")

#     # Use the policy network to choose the best action
#     with torch.no_grad():
#         action_index = self.policy_net(state).argmax().item()
#     # action_index = self.policy_net(state).argmax().item()

#     # Map the action index to the corresponding action
#     ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#     self_action = ACTIONS[action_index]

#     # Return the action
#     return self_action

def act(self, game_state):
    # Extract features
    state = state_to_features(game_state)
    self.logger.info(f"Extracted features: {state}")

    # Convert to tensor
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    # Get Q-values
    with torch.no_grad():
        q_values = self.policy_net(state)

    # Choose action
    action_index = q_values.argmax().item()
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    self_action = ACTIONS[action_index]

    # Log decision
    self.logger.info(f"Q-values: {q_values.cpu().numpy()}, Chosen action: {self_action}")

    return self_action

def game_events_occurred(self, old_game_state, new_game_state, action, events):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    # Convert old_game_state and new_game_state to features
    state = state_to_features(old_game_state) if old_game_state is not None else None
    new_state = state_to_features(new_game_state) if new_game_state is not None else None
    
    # Calculate the reward based on events
    reward = calculate_reward(events)  # Ensure this function is defined correctly

    # Define whether the episode is done
    done = new_game_state is None

    # Add the experience to the replay buffer
    if state is not None and new_state is not None:
        self.replay_buffer.append(state, action, reward, new_state, done)

def compute_reward(events, performance_measure):
    """
    Calculate the reward based on various game events, with added complexity.
    Dynamically adjust rewards based on the performance measure.
    """
    threshold = 100  # Example threshold, adjust based on your needs
    reward = 0
    
    # Positive rewards
    if 'COIN_COLLECTED' in events:
        reward += 10  # Increase reward for collecting coins
    if 'KILLED_OPPONENT' in events:
        reward += 30  # Higher reward for killing an opponent
    if 'CRATE_DESTROYED' in events:
        reward += 3  # Smaller reward for destroying a crate
    
    # Negative rewards
    if 'KILLED_SELF' in events:
        reward -= 30  # Increase penalty for self-destruction
    if 'GOT_KILLED' in events:
        reward -= 15  # Increase penalty for getting killed
    if 'BOMB_DROPPED' in events:
        reward -= 1.0  # Increase penalty if agent drops bombs carelessly
    
    # Additional penalty for self-destruction by bomb explosion
    if 'BOMB_EXPLODED' in events and 'SELF_DESTROYED' in events:
        reward -= 20  # Heavy penalty for self-destruction
    
    # Penalty for staying near bombs or repeating the same action too often
    if 'NEAR_BOMB' in events:
        reward -= 10  # Increased penalty to encourage avoiding bombs
    if 'REPEATED_ACTION' in events:
        reward -= 5
    
    # Dynamic adjustments based on performance measure
    if performance_measure < threshold:
        # If performance is below threshold, increase the penalty for negative events
        if 'KILLED_SELF' in events:
            reward -= 10  # Additional penalty for poor performance
        if 'GOT_KILLED' in events:
            reward -= 10  # Additional penalty for getting killed

    # Encourage survival by penalizing risky behavior when the agent is underperforming
    if performance_measure < threshold and 'NEAR_BOMB' in events:
        reward -= 5  # Additional penalty for being near a bomb when performance is low

    return reward


def end_of_round(self, last_game_state, last_action, events):
    """Update the model after the round ends."""
    old_state = state_to_features(last_game_state)
    action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'BOMB': 4, 'WAIT': 5}
    action_idx = action_mapping.get(last_action, -1)  # Ensure action is an integer
    current_performance = calculate_performance(self.agent.rewards_history)
    reward = compute_reward(events, current_performance)
    done = 1
    self.agent.store_transition(old_state, action_idx, reward, None, done)
    self.agent.experience_replay()
    self.agent.update_target_network()