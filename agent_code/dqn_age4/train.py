
import torch
import torch.optim as optim
from .dqn_model import DQNAgent
from .features import state_to_features

def setup_training(self):
    """Initialize anything needed specifically for training."""
    self.agent.exploration_rate = 1.0  # Reset exploration rate for fresh training
    self.logger.info("Training setup complete.")
def is_in_bomb_range(agent_position, bomb_positions, bomb_range=3):
    """
    Check if the agent is within the range of any bombs.
    
    Args:
        agent_position (tuple)): The (x, y)) position of the agent.
        bomb_positions (list of tuples)): List of bomb positions.
        bomb_range (int)): The blast radius of the bombs.
        
    Returns:
        bool: True if the agent is in the blast radius of any bomb, False otherwise.
    """
    agent_x, agent_y = agent_position

    for bomb_position in bomb_positions:
        bomb_x, bomb_y = bomb_position
        
        # Check if the agent is in the same row or column as the bomb and within the blast radius
        if agent_x == bomb_x and abs(agent_y - bomb_y) <= bomb_range:
            return True
        if agent_y == bomb_y and abs(agent_x - bomb_x) <= bomb_range:
            return True

    return False


def moved_to_safe_spot(old_game_state, new_game_state):
    """
    Checks if the agent has moved to a safe spot after a bomb was placed.
    A safe spot is considered to be out of the bomb's blast radius.
    """
    bombs = old_game_state['bombs']
    self_position = new_game_state['self'][3]  # New position of the agent
    
    if not bombs:
        return True  # Safe if no bombs

    for (bomb_position, timer) in bombs:
        if is_in_bomb_range(self_position, [bomb_position]):  # Use is_in_bomb_range
            return False  # Still in bomb range, not safe
    
    return True  # Successfully moved to a safe spot



def calculate_performance(rewards_history):
    """Calculate the moving average of the last N rewards."""
    N = 100  # Smooth over the last 100 rewards
    if len(rewards_history) < N:
        return sum(rewards_history) / len(rewards_history) if len(rewards_history) > 0 else 0
    return sum(rewards_history[-N:]) / N

def compute_reward(events, performance_measure, threshold=100, additional_penalty=5):
    """Calculate the reward based on game events with dynamic adjustments."""
    reward = 0
    if 'COIN_COLLECTED' in events:
        reward += 1
    if 'KILLED_OPPONENT' in events:
        reward += 5
    if 'KILLED_SELF' in events:
        reward -= 10
    if 'GOT_KILLED' in events:
        reward -= 5

    # Dynamically adjust rewards based on performance
    if performance_measure < threshold:
        reward -= additional_penalty  # Penalize further if underperforming

    return reward
def determine_done_condition(events):
    """
    Determines if the episode is done based on game events.
    Typically, an episode is done if the agent dies, or the game ends.
    """
    # Check if any of the end-game conditions are in the events
    if 'KILLED_SELF' in events or 'GOT_KILLED' in events or 'END_GAME' in events:
        return True  # The episode is done
    return False  # The episode continues

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # Convert the old and new game states to feature vectors (states)
    if old_game_state is not None:
        state = state_to_features(old_game_state)
    else:
        state = None

    if new_game_state is not None:
        new_state = state_to_features(new_game_state)
    else:
        new_state = None

    action = self_action  # Already provided

    # Compute the performance measure based on the agent's reward history
    performance_measure = sum(self.agent.rewards_history) / len(self.agent.rewards_history) if self.agent.rewards_history else 0

    # Compute the reward based on the events and performance measure
    reward = compute_reward(events, performance_measure)

    # Adjust reward if agent moved to a safe spot
    if moved_to_safe_spot(old_game_state, new_game_state):
        reward += 10  # Extra reward for moving to a safe spot

    # Determine if the episode is done
    done = determine_done_condition(events)

    # Add experience to the replay buffer (use `push`)
    self.replay_buffer.push(state, action, reward, new_state, done)

def end_of_round(self, last_game_state, last_action, events):
    """Finalize updates or exploration decay at the end of each round."""
    old_state = state_to_features(last_game_state)
    action_idx = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'].index(last_action)
    
    last_reward = self.agent.rewards_history[-1] if self.agent.rewards_history else 0

    # Compute the reward for the current round using the latest reward for reference
    reward = compute_reward(events, last_reward)

    # Store the transition in the replay buffer
    self.agent.store_transition(old_state, action_idx, reward, None, 1)  # 'done' is true as the round ends
    
    # Perform experience replay and update target network
    self.agent.experience_replay()
    self.agent.update_target_network()
    
    # Optionally, decay the exploration rate
    self.agent.update_exploration_rate()

    self.agent.save_model("latest_model.pth")

def main_training_loop():
    self.agent.rewards_history = []  # Clear rewards history

    for episode in range(total_episodes):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        self.agent.rewards_history.append(total_reward)

        # Update target network every 5 episodes
        if episode % 5 == 0:
            self.agent.update_target_network()

        if episode % 10 == 0:
            current_performance = calculate_performance(self.agent.rewards_history)
            print(f'Episode {episode}, Performance: {current_performance}')
