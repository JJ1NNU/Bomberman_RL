"""
<<CowardAgent 규칙 설명>>
    1. 즉각적인 생존: 폭발 위험이 있으면 무조건 탈출 (최우선).
    2. 공포 반응 (Panic): 반경 6칸 이내에 적이 있다면? → 적에게서 멀어지는 방향으로 전력 질주.
    3. 구석 숨기 (Hiding): 적이 멀리 있다면? → 맵의 **구석(Corner)**이나 가장자리로 이동해서 숨음.
    4. 평화주의: 폭탄은 절대로 사용하지 않음 (오폭 사고 방지).
"""

import numpy as np
from random import shuffle
from collections import deque
import settings as s

# 절대 공격하지 않음
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Coward agent: ready to run away.')
    np.random.seed()

def act(self, game_state):
    """
    겁쟁이 에이전트 행동 로직:
    1. 폭발 위험 감지 시 탈출
    2. 적이 근처(6칸 이내)에 있으면 도망
    3. 안전하면 구석으로 이동하여 숨기
    """
    
    # 1. 정보 수집
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosions = game_state['explosion_map']

    # 2. 위험 지도 생성 (Danger Map)
    danger_map = np.zeros(arena.shape)
    danger_map[explosions > 0] = 1 
    
    for (bx, by), t in bombs:
        danger_map[bx, by] = 1
        for i in range(1, s.BOMB_POWER + 1):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = bx + (dx * i), by + (dy * i)
                if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]): continue
                if arena[nx, ny] == -1: break 
                danger_map[nx, ny] = 1

    # --- 행동 결정 로직 ---

    # Priority 1: 생존 (폭탄/폭발 회피)
    if danger_map[x, y] == 1:
        self.logger.debug("BOMB DANGER! Escaping...")
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        return best_escape_move if best_escape_move else 'WAIT'

    # Priority 2: 공포 (적 회피)
    # 가장 가까운 적과의 거리 계산
    nearest_enemy, dist = get_nearest_enemy_dist((x, y), others)
    
    # 적이 '공포 반경(6칸)' 안에 들어오면 도망
    if nearest_enemy and dist < 6:
        self.logger.debug(f"Enemy within {dist} tiles! Running away...")
        # 적에게서 멀어지는 방향으로 이동
        run_move = get_move_away_from_enemy(arena, danger_map, (x, y), nearest_enemy, others, bomb_xys)
        if run_move:
            return run_move

    # Priority 3: 숨기 (Hiding)
    # 적이 멀리 있으면 구석(안전지대)을 찾아서 이동
    # 맵의 네 모서리 중 안전하고 갈 수 있는 곳을 타겟으로 잡음
    corners = [(1, 1), (1, arena.shape[1]-2), (arena.shape[0]-2, 1), (arena.shape[0]-2, arena.shape[1]-2)]
    safe_corners = [c for c in corners if arena[c] == 0 and danger_map[c] == 0 and c not in others]
    
    if safe_corners:
        # 가장 가까운 구석으로 이동
        move = get_move_to_nearest_target(arena, danger_map, (x, y), safe_corners, others, bomb_xys)
        if move:
             return move

    # 아무것도 할 게 없거나 구석에 이미 있으면 대기 (혹은 소심하게 랜덤 이동)
    return np.random.choice(['WAIT', 'UP', 'DOWN', 'LEFT', 'RIGHT'], p=[0.6, 0.1, 0.1, 0.1, 0.1])


# --- Helper Functions ---

def get_nearest_enemy_dist(my_pos, others):
    """가장 가까운 적과 그 거리(Manhattan Distance) 반환"""
    if not others:
        return None, 999
    
    x, y = my_pos
    nearest = min(others, key=lambda o: abs(x - o[0]) + abs(y - o[1]))
    dist = abs(x - nearest[0]) + abs(y - nearest[1])
    return nearest, dist

def get_move_away_from_enemy(arena, danger_map, my_pos, enemy_pos, others, bomb_xys):
    """
    적의 위치로부터 거리가 멀어지는 방향으로 이동.
    단순 Greedy 방식: 갈 수 있는 칸 중 적과의 거리가 가장 먼 칸 선택.
    """
    x, y = my_pos
    ex, ey = enemy_pos
    current_dist = abs(x - ex) + abs(y - ey)
    
    best_move = None
    max_dist = -1
    
    directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
    shuffle(directions) # 동일 조건 시 랜덤성 부여
    
    for action, (dx, dy) in directions:
        nx, ny = x + dx, y + dy
        
        # 이동 가능 조건 (벽X, 폭탄X, 다른사람X, 위험지역X)
        if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
            arena[nx, ny] == 0 and (nx, ny) not in others and (nx, ny) not in bomb_xys and 
            danger_map[nx, ny] == 0):
            
            # 적과의 새로운 거리 계산
            new_dist = abs(nx - ex) + abs(ny - ey)
            
            # 거리가 멀어지면(혹은 유지되면) 후보 등록
            if new_dist > current_dist:
                return action # 바로 도망
            
            # 만약 바로 멀어지는 길이 막혔다면, 적어도 거리가 덜 줄어드는 곳(혹은 최선)을 찾기 위해 기록
            if new_dist > max_dist:
                max_dist = new_dist
                best_move = action
                
    return best_move

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """(기존과 동일) 위험지역 탈출"""
    queue = deque([(start, None)])
    visited = set([start])
    while queue:
        (cx, cy), first_action = queue.popleft()
        if danger_map[cx, cy] == 0: return first_action
        
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and (nx, ny) not in others and (nx, ny) not in bomb_xys and 
                (nx, ny) not in visited):
                visited.add((nx, ny))
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None

def get_move_to_nearest_target(arena, danger_map, start, targets, others, bomb_xys):
    """(기존과 동일) 특정 타겟(구석 등)으로 이동"""
    if not targets: return None
    queue = deque([(start, None)])
    visited = set([start])
    while queue:
        (cx, cy), first_action = queue.popleft()
        if (cx, cy) in targets: return first_action
        
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and (nx, ny) not in others and (nx, ny) not in bomb_xys and 
                danger_map[nx, ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None