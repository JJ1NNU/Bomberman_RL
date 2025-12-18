import numpy as np
from random import shuffle

def moved_to_safe_spot(game_state):
    # Logic to calculate safe spots away from the bomb's blast range
    danger_zone, urgent_danger_zone = calculate_danger_zone(game_state['bombs'])
    
    # Prioritize urgent danger zones
    safe_spot = find_nearest_safe_spot(game_state['self'][3], urgent_danger_zone, game_state['free_space'])
    
    if not safe_spot:  # If no urgent danger, find general safe spot
        safe_spot = find_nearest_safe_spot(game_state['self'][3], danger_zone, game_state['free_space'])

    # Assuming you have a function to decide which direction to move toward the safe spot
    action_to_safe_spot = decide_direction_to_move(game_state['self'][3], safe_spot)

    return action_to_safe_spot


def calculate_danger_zone(bombs, blast_radius=3):
    danger_zone = set()
    urgent_danger_zone = set()  # Track areas where bombs will explode soon

    for bomb in bombs:
        bomb_position, timer = bomb[0], bomb[1]  # Unpack bomb position and timer
        
        x, y = bomb_position
        # Add positions within the blast radius to the danger zone
        for i in range(-blast_radius, blast_radius + 1):
            if 0 <= x + i < 15:  # Assuming the arena size is 15x15
                danger_zone.add((x + i, y))
                if timer < 3:  # If bomb will explode soon, mark it as urgent
                    urgent_danger_zone.add((x + i, y))
            if 0 <= y + i < 15:
                danger_zone.add((x, y + i))
                if timer < 3:
                    urgent_danger_zone.add((x, y + i))

    return danger_zone, urgent_danger_zone  # Return both general and urgent danger zones





def calculate_free_space(arena):
    """
    Calculate the free space in the arena.
    Free space is considered any tile that is not a wall (0).
    """
    free_space = np.zeros_like(arena, dtype=bool)
    free_space[arena == 0] = True  # Assuming 0 represents free space
    return free_space


def look_for_targets(free_space, start, targets):
    """
    Find the shortest path to one of the targets.
    This function performs a breadth-first search (BFS).
    """
    from collections import deque
    
    start = tuple(start)  # Convert the start position to a tuple
    targets = {tuple(t) for t in targets}  # Convert all target positions to tuples
    queue = deque([(start, 0)])  # Queue stores tuples of (position, distance)
    visited = {start}  # Use a set to track visited positions, and start with the starting position

    while queue:
        position, distance = queue.popleft()
        
        # If this position is one of the targets, return it
        if position in targets:
            return position
        
        # Explore neighboring positions (up, down, left, right)
        x, y = position
        neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        for neighbor in neighbors:
            # Ensure neighbor is within bounds and is free space
            if 0 <= neighbor[0] < free_space.shape[0] and 0 <= neighbor[1] < free_space.shape[1]:
                if free_space[neighbor] and neighbor not in visited:
                    visited.add(neighbor)  # Mark as visited
                    queue.append((neighbor, distance + 1))
    
    return None  # If no target is reachable

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def get_directions(agent_position):
    x, y = agent_position
    return {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y),
        'RIGHT': (x + 1, y),
        'BOMB': agent_position,
        'WAIT': agent_position
    }

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def calculate_free_space(arena):
    """
    Calculate the free space in the arena.
    Free space is considered any tile that is not a wall (0).
    """
    free_space = np.zeros_like(arena, dtype=bool)
    free_space[arena == 0] = True  # Assuming 0 represents free space
    return free_space
def calculate_bomb_map(arena, bombs, explosion_timer=4):
    bomb_map = np.zeros_like(arena, dtype=int)
    for bomb in bombs:
        if isinstance(bomb[0], tuple):
            bomb_position, timer = bomb  # Unpack the tuple as expected
        else:
            bomb_position = bomb  # Assuming bomb is just the position (x, y)
            timer = explosion_timer  # Default timer if not provided
        
        x, y = bomb_position  # Unpack the bomb position
        bomb_map[x, y] = timer

        # Spread the timer to adjacent positions within blast radius
        for i in range(1, explosion_timer):
            if x - i >= 0:
                bomb_map[x - i, y] = max(bomb_map[x - i, y], timer)
            if x + i < bomb_map.shape[0]:
                bomb_map[x + i, y] = max(bomb_map[x + i, y], timer)
            if y - i >= 0:
                bomb_map[x, y - i] = max(bomb_map[x, y - i], timer)
            if y + i < bomb_map.shape[1]:
                bomb_map[x, y + i] = max(bomb_map[x, y + i], timer)
    return bomb_map

def state_to_features(game_state: dict):
    if game_state is None:
        return np.zeros(350)  # Adjusted for the additional features

    # Existing features
    field = game_state['field'].flatten()  # Flatten the game field (2D to 1D)
    own_position = np.array(game_state['self'][3])  # Extract own position (x, y)
    free_space = calculate_free_space(game_state['field'])  # Calculate free space
    
    # Calculate danger zones based on bomb positions and timers
    bombs = game_state['bombs']  # Extract bomb positions
    danger_zone, urgent_danger_zone = calculate_danger_zone(bombs)
    bomb_map = calculate_bomb_map(game_state['field'], bombs)
    
    # Extract features
    features = []
    features.extend(feature1(game_state, free_space, own_position))  # Target-related features
    features.extend(feature2(game_state, danger_zone, urgent_danger_zone, bomb_map))  # Bomb danger features
    features.extend(feature3(game_state, danger_zone, urgent_danger_zone))  # General danger features

    # Concatenate all features
    features = np.concatenate((field, own_position, features))

    # Ensure features vector has exactly 350 elements
    if len(features) < 350:
        features = np.pad(features, (0, 350 - len(features)), 'constant')
    elif len(features) > 350:
        features = features[:350]

    return features


def feature1(game_state, free_space, agent_position):
    feature = []
    best_direction = look_for_targets(free_space, agent_position, game_state['coins'])
    directions = get_directions(agent_position)

    for action in ACTIONS:
        new_position = directions[action]
        if np.array_equal(new_position, agent_position):
            feature.append(0)
        elif np.array_equal(new_position, best_direction):
            feature.append(1)
        else:
            feature.append(0)
    return feature

def feature2(game_state, danger_zone, urgent_danger_zone, bomb_map):
    feature = []
    agent_position = game_state['self'][3]
    directions = get_directions(agent_position)

    for action in ACTIONS:
        new_position = directions[action]
        if new_position in urgent_danger_zone:  # If position is in an urgent danger zone
            feature.append(1)  # High danger, high priority to avoid
        elif new_position in danger_zone and bomb_map[new_position] == 0:
            feature.append(1)  # General danger, move away
        else:
            feature.append(0)  # Safe position
    return feature

def feature3(game_state, danger_zone, urgent_danger_zone):
    feature = []
    agent_position = game_state['self'][3]
    directions = get_directions(agent_position)

    for action in ACTIONS:
        new_position = directions[action]
        if new_position in urgent_danger_zone:  # Urgent danger zone
            feature.append(1)  # Highest priority to avoid
        elif new_position in danger_zone:
            feature.append(1)  # General danger
        else:
            feature.append(0)  # Safe
    return feature
