"""
<< DestroyerAgent 규칙 설명 >>
    1. 생존 우선: 폭발 범위에 있으면 무조건 도망갑니다. (Peaceful과 동일)
    2. 타겟 설정: 맵의 모든 **"상자(Crate) 옆 빈칸"**이 목적지가 됩니다.
    3. 폭파: 상자 옆에 도착했고, "폭탄을 놔도 도망갈 구석이 있다면" 폭탄을 설치합니다.
    4. 무시: 코인이나 적은 신경 쓰지 않습니다. (상자가 다 깨지면 그제야 코인을 줍거나 랜덤 이동합니다)
"""

import numpy as np
from random import shuffle
from collections import deque
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

def act(self, game_state):
    """
    파괴자 에이전트 행동 로직:
    1. 위험 감지 및 탈출
    2. 폭탄 설치 (상자 옆이고 안전할 때)
    3. 상자 추적 (가장 가까운 상자 옆으로 이동)
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
    danger_map[explosions > 0] = 1 # 현재 폭발 중
    
    # 곧 터질 폭탄 범위 예측
    for (bx, by), t in bombs:
        danger_map[bx, by] = 1
        for i in range(1, s.BOMB_POWER + 1):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = bx + (dx * i), by + (dy * i)
                if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]): continue
                if arena[nx, ny] == -1: break 
                danger_map[nx, ny] = 1

    # --- 행동 결정 로직 ---

    # Priority 1: 생존 (위험지역 탈출)
    if danger_map[x, y] == 1:
        self.logger.debug("DANGER! Escaping...")
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        return best_escape_move if best_escape_move else 'WAIT'

    # Priority 2: 폭탄 설치 (상자 파괴)
    # 조건: 폭탄이 있고, 상자 옆이고, 폭탄을 놔도 살 수 있어야 함
    if bombs_left > 0:
        if is_adjacent_to_crate(arena, (x, y)):
            # 중요: 내가 지금 폭탄을 놨다고 가정했을 때 도망갈 길이 있는가?
            if is_safe_to_place_bomb(arena, danger_map, (x, y), others, bomb_xys):
                self.logger.debug("Placing bomb to destroy crate!")
                return 'BOMB'

    # Priority 3: 상자 사냥 (가장 가까운 상자 옆 타일로 이동)
    # 상자 옆에 설 수 있는 모든 좌표를 타겟으로 설정
    crate_targets = get_crate_targets(arena, others, bomb_xys, danger_map)
    
    if crate_targets:
        self.logger.debug("Moving towards nearest crate...")
        move = get_move_to_nearest_target(arena, danger_map, (x, y), crate_targets, others, bomb_xys)
        if move:
            return move

    # Priority 4: 상자가 다 없어졌으면? (할 일 없음 -> 코인이라도 줍자)
    if coins:
        move = get_move_to_nearest_target(arena, danger_map, (x, y), coins, others, bomb_xys)
        if move:
            return move

    # 할 거 없으면 랜덤 배회
    return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])


# --- Helper Functions ---

def is_adjacent_to_crate(arena, pos):
    """현재 위치 상하좌우에 상자가 있는지 확인"""
    x, y = pos
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
            if arena[nx, ny] == 1: # 1 is crate
                return True
    return False

def is_safe_to_place_bomb(arena, current_danger_map, pos, others, bomb_xys):
    """
    내가 여기에 폭탄을 놨다고 가정(Simulation)했을 때, 
    폭발하기 전에 안전지대로 도망갈 수 있는지 확인 (자살 방지)
    """
    x, y = pos
    
    # 가상의 위험 지도 생성
    simulated_danger = current_danger_map.copy()
    simulated_danger[x, y] = 1
    for i in range(1, s.BOMB_POWER + 1):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + (dx * i), y + (dy * i)
            if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]): continue
            if arena[nx, ny] == -1: break 
            simulated_danger[nx, ny] = 1
            
    # 가상의 위험 지도에서 탈출구가 있는지 BFS로 확인
    # (폭탄이 터지기까지 4턴 여유가 있다고 가정하고, 4턴 내에 안전지대 도달 가능한지 체크)
    # 간단하게는 그냥 '안전지대 경로가 존재하느냐'만 봐도 충분함
    start_node = (x, y)
    queue = deque([start_node])
    visited = set([start_node])
    
    while queue:
        cx, cy = queue.popleft()
        
        # 안전한 곳을 찾았다! (현재 위험하지도 않고, 내 폭탄에도 안 맞는 곳)
        if simulated_danger[cx, cy] == 0:
            return True
            
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and 
                (nx, ny) not in others and 
                (nx, ny) not in bomb_xys and # 기존 폭탄들
                (nx, ny) != (x, y) and # 방금 놓은 내 폭탄 위치 제외
                (nx, ny) not in visited):
                
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return False # 도망갈 곳이 없음 (갇힘)

def get_crate_targets(arena, others, bomb_xys, danger_map):
    """상자 바로 옆의 빈칸(접근 가능한 타일)들을 리스트로 반환"""
    targets = []
    rows, cols = arena.shape
    for x in range(rows):
        for y in range(cols):
            # 빈칸이고, 위험하지 않고, 다른 사람이 없고, 폭탄이 없는 곳
            if (arena[x, y] == 0 and danger_map[x, y] == 0 and 
                (x, y) not in others and (x, y) not in bomb_xys):
                # 상하좌우 중 하나라도 상자(1)가 있으면 타겟임
                if is_adjacent_to_crate(arena, (x, y)):
                    targets.append((x, y))
    return targets

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """(Peaceful Agent와 동일) 가장 가까운 안전지대로 탈출"""
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
    """(Peaceful Agent와 동일) 타겟 리스트 중 가장 가까운 곳으로 이동"""
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