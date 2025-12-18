"""
<<LazyAgent 규칙 설명>>
    1. 생존 본능: 폭탄이 날아오면 피합니다 (이건 기본).
    2. 캠핑 장소 탐색: 맵에서 가장 가까운 **"숨기 좋은 곳"**을 찾습니다.
        1순위: 막다른 길 (3면이 막힌 곳) - 폭탄 피하기가 어렵지만, 남들이 굳이 들어와서 죽이지 않는 이상 안전함.
        2순위: 구석 (Corner) - 폭발에 휘말릴 확률이 가장 낮음.
    3. 은신 (Camping): 캠핑 장소에 도착하면, 아무것도 하지 않고 **WAIT**만 계속 보냅니다.
    4. 무관심: 코인, 상자, 적 모두 무시합니다.
"""

import numpy as np
from random import shuffle
from collections import deque
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Lazy agent: I just want to sleep.')
    np.random.seed()
    
    # 이 에이전트는 한 번 정한 캠핑 장소를 기억합니다.
    self.camp_site = None

def act(self, game_state):
    """
    게으른 에이전트 행동 로직:
    1. 위험하면 피한다.
    2. 안전하면 쉰다 (WAIT).
    3. 쉬기 불편하면 숨을 곳(구석/막다른길)을 찾아간다.
    """
    
    # 1. 정보 수집
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    explosions = game_state['explosion_map']

    # 2. 위험 지도 생성
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
        self.logger.debug("Too loud here! Moving to safety...")
        # 캠핑이고 뭐고 일단 살아야 함
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        return best_escape_move if best_escape_move else 'WAIT'

    # Priority 2: 캠핑 (이미 좋은 자리에 있다면 꼼짝도 안 함)
    if is_good_camp_spot(arena, (x, y)):
        self.logger.debug("Zzz... (Waiting)")
        # 위험하지 않다면 무조건 대기
        return 'WAIT'

    # Priority 3: 캠핑 장소 찾기 (좋은 자리가 아니라면 이동)
    self.logger.debug("Looking for a bed...")
    
    # 캠핑 후보지: 막다른 길(Dead ends) 또는 구석(Corners)
    camping_spots = find_camping_spots(arena, others, bomb_xys, danger_map)
    
    if camping_spots:
        # 가장 가까운 캠핑 장소로 이동
        move = get_move_to_nearest_target(arena, danger_map, (x, y), camping_spots, others, bomb_xys)
        if move:
            return move

    # 갈 곳도 없고 위험하지도 않으면 그냥 제자리 대기
    return 'WAIT'


# --- Helper Functions ---

def is_good_camp_spot(arena, pos):
    """
    현재 위치가 숨기 좋은 곳인지 판단.
    기준: 4면 중 3면 이상이 막혀있거나(막다른 길), 2면이 막힌 구석.
    """
    x, y = pos
    blocked_count = 0
    # 상하좌우 체크
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        # 맵 밖이거나, 벽이거나, 상자면 '막힌 것'으로 간주
        if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
            blocked_count += 1
        elif arena[nx, ny] != 0: # -1(돌벽) or 1(상자)
            blocked_count += 1
            
    # 3면 이상 막히면(막다른 길) 최고, 2면 이상(구석/복도)도 나쁘지 않음
    # 여기서는 '은신' 느낌을 위해 3면 이상(Dead end)을 선호하되, 없으면 구석도 OK
    return blocked_count >= 3

def find_camping_spots(arena, others, bomb_xys, danger_map):
    """맵 전체에서 숨기 좋은 좌표들을 리스트로 반환"""
    spots = []
    rows, cols = arena.shape
    for x in range(rows):
        for y in range(cols):
            # 빈 땅이고, 위험하지 않고, 사람이 없고, 폭탄이 없는 곳
            if (arena[x, y] == 0 and danger_map[x, y] == 0 and 
                (x, y) not in others and (x, y) not in bomb_xys):
                
                if is_good_camp_spot(arena, (x, y)):
                    spots.append((x, y))
                    
    # 만약 완벽한 은신처(3면 막힘)가 없다면, 구석(2면 막힘)이라도 찾음
    if not spots:
        for x in range(rows):
            for y in range(cols):
                if (arena[x, y] == 0 and danger_map[x, y] == 0 and 
                    (x, y) not in others and (x, y) not in bomb_xys):
                    # 구석 체크 (blocked >= 2)
                    blocked_count = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < rows and 0 <= ny < cols) or arena[nx, ny] != 0:
                            blocked_count += 1
                    if blocked_count >= 2:
                        spots.append((x, y))
    return spots

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """(기존과 동일)"""
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
    """(기존과 동일)"""
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