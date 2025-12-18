"""
<<HunterAgent 규칙 설명>>
    1. 생존 본능: 폭탄이 터질 위기라면 무조건 도망갑니다. (이건 모든 고성능 에이전트 공통)
    2. 킬각 측정 (Kill Shot): 적이 내 폭발 범위(직선 3칸) 안에 있고, 중간에 벽이 없다면? → 즉시 폭탄 설치.
    3. 근접 함정 (Trap): 적과 아주 가까이(1~2칸) 붙었다면? → 폭탄 설치 (가두기 시도).
    4. 추격 (Chase): 킬각이 안 나오면, 맵에서 가장 가까운 적을 향해 최단 경로로 달려갑니다.
"""

import numpy as np
from random import shuffle
from collections import deque
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Hunter agent ready to hunt.')
    np.random.seed()

def act(self, game_state):
    """
    살인마 에이전트 행동 로직:
    1. 위험 감지 및 회피 (죽으면 킬도 못 하니까)
    2. 공격 (적이 사정거리에 있거나 매우 근접했을 때)
    3. 추격 (가장 가까운 적에게 이동)
    """
    
    # 1. 정보 수집
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
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

    # Priority 1: 생존 (위험지역 탈출)
    if danger_map[x, y] == 1:
        self.logger.debug("DANGER! Escaping...")
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        return best_escape_move if best_escape_move else 'WAIT'

    # Priority 2: 공격 (Attack)
    if bombs_left > 0:
        # 적이 내 공격 범위에 들어왔는가?
        target_enemy = check_kill_opportunity(arena, (x, y), others)
        
        # 기회다! (혹은 적과 초근접 상태다)
        if target_enemy or is_enemy_too_close(x, y, others):
            # 중요: 공격하다 자폭하면 안 됨. 퇴로 확인.
            if is_safe_to_place_bomb(arena, danger_map, (x, y), others, bomb_xys):
                self.logger.debug(f"Placing bomb to KILL {target_enemy}!")
                return 'BOMB'

    # Priority 3: 추격 (Chase)
    # 가장 가까운 적을 찾아서 이동
    if others:
        self.logger.debug("Chasing enemy...")
        move = get_move_to_nearest_enemy(arena, danger_map, (x, y), others, bomb_xys)
        if move:
            return move

    # 적이 없거나(승리), 갈 길이 없으면 랜덤 배회
    return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'])


# --- Helper Functions ---

def check_kill_opportunity(arena, my_pos, others):
    """
    내 위치(my_pos)에서 폭탄을 놓았을 때 죽일 수 있는 적이 있는지 확인.
    (직선상에 있고, 사거리 내에 있고, 벽으로 안 막혀 있어야 함)
    """
    x, y = my_pos
    for (ox, oy) in others:
        if x == ox: # 같은 세로줄
            dist = abs(y - oy)
            if dist <= s.BOMB_POWER:
                # 중간에 벽이 있는지 확인
                step = 1 if oy > y else -1
                blocked = False
                for k in range(1, dist):
                    if arena[x, y + k*step] == -1: # 돌벽은 막힘 (상자는 뚫림)
                        blocked = True
                        break
                if not blocked: return (ox, oy)
        
        elif y == oy: # 같은 가로줄
            dist = abs(x - ox)
            if dist <= s.BOMB_POWER:
                step = 1 if ox > x else -1
                blocked = False
                for k in range(1, dist):
                    if arena[x + k*step, y] == -1:
                        blocked = True
                        break
                if not blocked: return (ox, oy)
    return None

def is_enemy_too_close(x, y, others):
    """적이 바로 옆(1칸 거리)에 붙었는지 확인 (함정 설치용)"""
    for (ox, oy) in others:
        if abs(x - ox) + abs(y - oy) <= 1:
            return True
    return False

def get_move_to_nearest_enemy(arena, danger_map, start, others, bomb_xys):
    """가장 가까운 적에게 다가가는 최단 경로 (BFS)"""
    # 적들의 위치를 목표 지점으로 설정
    queue = deque([(start, None)])
    visited = set([start])
    
    while queue:
        (cx, cy), first_action = queue.popleft()
        
        # 적의 위치에 도달했거나, 적 바로 옆칸에 도착했으면 성공
        if (cx, cy) in others:
            return first_action
            
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            # 이동 가능 조건 (벽X, 폭탄X, 위험지역X)
            # 주의: Hunter는 적을 '통과'할 수 없으므로 others 체크는 뺌(목적지니까)
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and (nx, ny) not in bomb_xys and 
                danger_map[nx, ny] == 0 and (nx, ny) not in visited):
                
                visited.add((nx, ny))
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None

def is_safe_to_place_bomb(arena, current_danger_map, pos, others, bomb_xys):
    """(Destroyer와 동일) 폭탄 설치 후 생존 가능성 시뮬레이션"""
    x, y = pos
    simulated_danger = current_danger_map.copy()
    simulated_danger[x, y] = 1
    for i in range(1, s.BOMB_POWER + 1):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + (dx * i), y + (dy * i)
            if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]): continue
            if arena[nx, ny] == -1: break 
            simulated_danger[nx, ny] = 1
            
    queue = deque([(x, y)])
    visited = set([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        if simulated_danger[cx, cy] == 0: return True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and (nx, ny) not in bomb_xys and (nx, ny) != (x, y) and 
                (nx, ny) not in visited and (nx, ny) not in others):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """(Peaceful Agent와 동일)"""
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