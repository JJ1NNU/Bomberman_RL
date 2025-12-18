"""
<<KamikazaeAgent 규칙 설명>>
    1. 목표 설정: 맵의 모든 적 중에서 가장 가까운 적을 타겟으로 삼습니다.
    2. 맹추격 (Aggressive Chase): 안전보다는 속도입니다. 적에게 가는 최단 경로로 돌진합니다.
    3. 자폭 공격 (Suicide Bombing):
        적이 내 폭발 범위(3칸) 안에 있고, 벽이 없다면? → 즉시 폭탄 설치.
        중요: 내가 도망갈 곳이 있는지(Survival Check)는 확인하지 않습니다.
    4. 동귀어진: 적과 딱 붙었을 때(거리 1) 가장 폭탄을 잘 놓습니다.
"""

import numpy as np
from random import shuffle
from collections import deque
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Kamikaze agent: BANZAI!')
    np.random.seed()

def act(self, game_state):
    """
    자폭맨 행동 로직:
    1. 적이 사정거리 내에 있으면 무조건 폭탄 (내 생존 여부 확인 안 함)
    2. 적이 없으면 적에게 돌진
    3. 폭탄이 없거나 쿨타임이면, 적에게 비비기 (진로 방해)
    """
    
    # 1. 정보 수집
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    explosions = game_state['explosion_map']

    # 2. 위험 지도 생성 (폭탄 피하기용 - 적을 만나기 전까진 살아야 하니까)
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

    # Priority 1: 자폭 공격 (Suicide Attack)
    # 적이 내 사정권에 들어왔는가?
    if bombs_left > 0:
        # Hunter와 달리 is_safe_to_place_bomb 체크를 하지 않음!
        if is_enemy_in_blast_range(arena, (x, y), others):
            self.logger.debug("Enemy spotted! KAMIKAZE ATTACK!")
            return 'BOMB'

    # Priority 2: 추격 (Chase)
    # 적에게 닥치고 돌진 (위험지역만 피해서)
    if others:
        self.logger.debug("Charging towards enemy...")
        # Kamikaze는 적과 '겹치는 것'도 목표로 함 (비비기)
        move = get_move_to_nearest_enemy(arena, danger_map, (x, y), others, bomb_xys)
        if move:
            return move

    # Priority 3: 생존 (적을 못 만났는데 폭탄이 터지려 할 때만 피함)
    if danger_map[x, y] == 1:
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        if best_escape_move:
            return best_escape_move

    # 할 거 없으면 랜덤
    return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])


# --- Helper Functions ---

def is_enemy_in_blast_range(arena, my_pos, others):
    """
    적이 내 폭발 범위(직선 3칸) 안에 있는지 확인.
    자신이 죽는지는 신경 쓰지 않음.
    """
    x, y = my_pos
    for (ox, oy) in others:
        # 1. 같은 위치(겹침)면 무조건 폭파
        if x == ox and y == oy:
            return True

        # 2. 세로 선상 확인
        if x == ox:
            dist = abs(y - oy)
            if dist <= s.BOMB_POWER:
                # 벽 체크
                step = 1 if oy > y else -1
                blocked = False
                for k in range(1, dist):
                    if arena[x, y + k*step] == -1: # 돌벽만 아니면 됨 (상자도 뚫고 죽임)
                        blocked = True
                        break
                if not blocked: return True
        
        # 3. 가로 선상 확인
        elif y == oy:
            dist = abs(x - ox)
            if dist <= s.BOMB_POWER:
                step = 1 if ox > x else -1
                blocked = False
                for k in range(1, dist):
                    if arena[x + k*step, y] == -1:
                        blocked = True
                        break
                if not blocked: return True
    return False

def get_move_to_nearest_enemy(arena, danger_map, start, others, bomb_xys):
    """
    가장 가까운 적에게 다가가는 최단 경로 (BFS).
    Hunter와 달리 적의 바로 옆이 아니라 적의 위치 그 자체를 목표로 함.
    """
    queue = deque([(start, None)])
    visited = set([start])
    
    while queue:
        (cx, cy), first_action = queue.popleft()
        
        # 적 위치 도달 (혹은 바로 옆)
        if (cx, cy) in others:
            return first_action
            
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            # 이동 가능 조건
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and (nx, ny) not in bomb_xys and 
                danger_map[nx, ny] == 0 and (nx, ny) not in visited):
                
                visited.add((nx, ny))
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """기본 회피 로직"""
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