import pickle
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt

import events as e


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.5
ALPHA = 0.9


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.visited_states = []
    self.rounds = 0
    self.stored_transition = []
    self.scores = []

    if self.eval_mode:
        self.kills = 0
        self.coins = 0
        self.deaths = 0
        self.crates = 0
        self.exp_dir = None


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.rounds += 1

    if self.eval_mode:
        self.scores.append(last_game_state['self'][1])
        if e.GOT_KILLED in events or e.KILLED_SELF in events:
            self.deaths += 1
        if self.rounds % 100 == 0:
            store_experiment_data(self)

        return


    if e.KILLED_SELF in events:
        print("SUICIDE")
    elif e.GOT_KILLED in events:
        print("DIED")
        # input()

    visited_states = len(self.Q_table)
    last_step = last_game_state['step']
    print(f"Game ended at step: {last_step}")
    print(f"Visited states: {visited_states}")
    user_score = last_game_state['self'][1]
    others_score = [agent[1] for agent in last_game_state["others"]]
    print(f"Score: {user_score}. Enemy scores: {others_score}")
    print("--------------------------------------------------------------------\n")
    self.visited_states.append(visited_states)
    self.scores.append(user_score)



    if self.rounds % 100 == 0:
        with open("Q_table.pkl", "wb") as f:
            pickle.dump(self.Q_table, f)
        with open("features.pkl", "wb") as f:
            pickle.dump(self.features, f)
        
        # with open("neighbours.pkl", "wb") as f:
        #     pickle.dump(self.neighbours, f)
        # with open("neighbour_distances.pkl", "wb") as f:
        #     pickle.dump(self.neighbour_distances, f)

    if self.rounds % 100 == 0:
        plt.plot(self.visited_states)
        plt.savefig("visited_states.png")
        plt.close()
        plt.plot(self.scores)
        avg_scores = np.convolve(np.array(self.scores), np.ones(50)/50, mode='valid')
        plt.plot(avg_scores)
        plt.yticks(range(16))
        plt.grid(axis='y')
        plt.savefig("scores.png")
        plt.close()


def update_q_table(self, old_id: int, action: int, new_id: int, reward: float) -> None:
    """
    Update your q-table based on your agent's experience
    """
    self.Q_table[old_id, action] += ALPHA * (reward + GAMMA * np.max(self.Q_table[new_id]) - self.Q_table[old_id, action])


def update_q_table_SARSA(self, SARSA_tuple: tuple[int, int, float, int, int]) -> None:
    """
    Update your q-table based on your agent's experience
    """
    old_id, action, reward, new_id, new_action = SARSA_tuple
    self.Q_table[old_id, action] += ALPHA * (reward + GAMMA * self.Q_table[new_id, new_action] - self.Q_table[old_id, action])


def collect_data(self, events: List[str]) -> None:
    if e.GOT_KILLED in events or e.KILLED_SELF in events:
        self.deaths += 1
    if e.CRATE_DESTROYED in events:
        self.crates += events.count(e.CRATE_DESTROYED)
    if e.COIN_COLLECTED in events:
        self.coins += 1
    if e.KILLED_OPPONENT in events:
        self.kills += events.count(e.KILLED_OPPONENT)
    

def store_experiment_data(self): 
    if self.exp_dir is None:
        self.exp_dir = "evaluation/experiment" + str(len(os.listdir("evaluation")))
        os.mkdir(self.exp_dir)
    # Training scores
    with open(f"{self.exp_dir}/data.txt", "w") as f:
        f.write(f"Tested agent for {self.rounds} rounds \n")
        f.write(f"Score: {np.array(self.scores).sum()}. Average: {np.array(self.scores).mean()}. Std: {np.array(self.scores).std()}\n")
        f.write(f"Kills: {self.kills}. Average: {self.kills/self.rounds}\n")
        f.write(f"Coins: {self.coins}. Average: {self.coins/self.rounds}\n")
        f.write(f"Deaths: {self.deaths} Average: {self.deaths/self.rounds}\n")
        f.write(f"Crates: {self.crates} Average: {self.crates/self.rounds}\n")
    # Score histogram
    plt.hist(self.scores, bins = max(self.scores)+1, range=(0, max(self.scores)+1))
    plt.savefig(f"{self.exp_dir}/scores.png")
    plt.close()
    np.savetxt(f"{self.exp_dir}/scores.txt", self.scores)
    #np.savetxt(f"{self.exp_dir}/bomb_positions.txt", self.bombs)

    # Features
    with open(f"{self.exp_dir}/features.txt", "w") as f:
        f.write(f"Q table has {len(self.Q_table)} different states \n")
        f.write(f"Feature weigths: {self.weights} \n")
        f.write(f"Feature shapes: \n")
        for i, feature in enumerate(self.features):
            f.write(f"Feature {i}: {feature[0].shape} \n")
