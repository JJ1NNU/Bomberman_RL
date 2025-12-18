import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import deque
import random
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)


class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        # Update the input dimension to 350 to match your feature size
        self.fc1 = nn.Linear(350, 256)  # input_dim = 350
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, action_dim)  # output to action space

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device=None, replay_buffer_capacity=10000, batch_size=64,
                 gamma=0.99, lr=0.001, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.rewards_history = []

        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state):
        if torch.rand(1).item() < self.exploration_rate:
            return torch.randint(0, self.action_dim, (1,)).item()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.policy_net(state).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5}
        
        # Convert action to integer if it's a string
        if isinstance(action, str):
            action = action_mapping.get(action, 0)  # Default to 0 if unknown action
        elif not isinstance(action, (int, np.integer)):
            action = 0  # Default to 0 for any other type
        
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.rewards_history.append(reward)

        # Check for None or empty state and replace with zero tensor
        self.next_state = next_state if next_state is not None and len(next_state) > 0 else np.zeros(self.state_dim)



    def experience_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # Define action mapping
        action_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5}
        # Ensure all actions are correctly mapped, converting np.str_ to int
        # actions = [action_mapping[str(a)] if isinstance(a, np.str_) else int(a) for a in actions]
        def action_to_index(a):
            if isinstance(a, (str, np.str_)):
                return action_mapping.get(str(a), 0)  # Default to 0 if unknown action
            elif isinstance(a, (int, np.integer)):
                return int(a)
            else:
                return 0  # Default to 0 for any other type
        
        actions = [action_to_index(a) for a in actions]
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q-learning update
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()



        
        # Convert actions to integer indices
        actions = [action_mapping.get(str(a), int(a)) if isinstance(a, (str, np.str_)) else int(a) for a in actions]


    def select_action(self, state):
        # Get the danger zone feature (whether the agent is in danger)
        is_in_danger = state[-1]  # Assume danger zone flag is the last feature

        # Ensure cooldown period for bomb placement
        if self.bomb_cooldown > 0:
            self.bomb_cooldown -= 1
            # Avoid dropping bombs during cooldown
            available_actions = [0, 1, 2, 3, 5]  # Exclude "BOMB" action (4)
        else:
            available_actions = [0, 1, 2, 3, 4, 5]  # All actions available

        # Prioritize movement when the agent is in danger
        if is_in_danger:
            available_actions = [0, 1, 2, 3]  # Only movement actions allowed, no BOMB or WAIT
            action = np.random.choice(available_actions)  # Randomly choose a movement action
        else:
            # Select action based on policy or exploration
            if np.random.rand() < self.exploration_rate:
                action = np.random.choice(available_actions)  # Exploration: random action
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state.to(self.device))
                    action = torch.argmax(q_values).item()  # Exploitation: choose best action

        # Set bomb cooldown if the chosen action is a bomb drop
        if action == 4:  # "BOMB"
            self.bomb_cooldown = 10  # Cooldown for 10 steps before another bomb can be placed

        # Check if the action is repeated
        if action == self.last_action:
            self.repeated_action_count += 1
        else:
            self.repeated_action_count = 0

        self.last_action = action

        # If the same action is repeated more than 5 times, randomize the next action
        if self.repeated_action_count > 5:
            action = np.random.choice([0, 1, 2, 3])  # Force the agent to move in a different direction

        return action

    def update_exploration_rate(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename):
        try:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'replay_buffer': self.replay_buffer
            }, filename)
            logging.info(f"Model successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save model to {filename}. Error: {str(e)}")

    def load_model(self, filename):
        try:
            if os.path.exists(filename):
                checkpoint = torch.load(filename, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.exploration_rate = checkpoint['exploration_rate']
                self.replay_buffer = checkpoint['replay_buffer']
                logging.info(f"Model successfully loaded from {filename}")
            else:
                logging.warning(f"No model found at {filename}. Starting with a new model.")
        except Exception as e:
            logging.error(f"Failed to load model from {filename}. Error: {str(e)}")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch from the replay buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Log the shapes for debugging purposes
        logging.info(f"States shape: {[np.shape(s) for s in states]}")
        logging.info(f"Next states shape: {[np.shape(ns) for ns in next_states]}")
        logging.info(f"Actions: {actions}")
        logging.info(f"Rewards: {rewards}")
        logging.info(f"Dones: {dones}")

        # Filter out any next_states that do not match the expected shape
        filtered_next_states = [ns if np.shape(ns) == (self.buffer[0][0].shape[0],) else np.zeros_like(self.buffer[0][0]) for ns in next_states]

        return np.array(states), np.array(actions), np.array(rewards), np.array(filtered_next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
