import numpy as np
from random import shuffle
from collections import deque
import settings as s

# 이 에이전트는 절대 폭탄을 쓰지 않습니다.
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    """에이전트 초기화"""
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

def act(self, game_state):
    """
    매 턴 행동 결정:
    1. 위험 감지 (폭탄, 폭발)
    2. 생존을 위한 회피 (가장 가까운 안전지대 탐색)
    3. 코인 수집 (가장 가까운 코인 탐색)
    4. 랜덤 이동 (할 게 없을 때)
    """
    
    # 1. 정보 수집
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs'] # [(x, y), countdown]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosions = game_state['explosion_map']

    # 2. 위험 지도 생성 (Danger Map)
    # 0: 안전, 1: 위험 (폭발 중이거나 곧 폭발함)
    danger_map = np.zeros(arena.shape)
    
    # 현재 폭발 중인 곳 표시
    danger_map[explosions > 0] = 1 
    
    # 곧 터질 폭탄 범위 예측 (십자가 형태)
    for (bx, by), t in bombs:
        # 폭탄 자체 위치
        danger_map[bx, by] = 1
        # 상하좌우 폭발 범위 계산
        for i in range(1, s.BOMB_POWER + 1):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = bx + (dx * i), by + (dy * i)
                # 맵 밖이거나 벽이면 폭발이 멈춤
                if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]): continue
                if arena[nx, ny] == -1: break 
                
                danger_map[nx, ny] = 1

    # 3. 이동 가능한 방향 찾기
    directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
    shuffle(directions) # 랜덤성을 줘서 루프 방지
    
    valid_moves = []
    safe_moves = [] # 당장 죽지 않는 움직임

    for action, (dx, dy) in directions:
        nx, ny = x + dx, y + dy
        # 맵 범위 체크 & 벽/상자 체크 & 다른 에이전트 체크 & 폭탄 위치 체크
        if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
            arena[nx, ny] == 0 and 
            (nx, ny) not in others and 
            (nx, ny) not in bomb_xys):
            
            valid_moves.append(action)
            
            # 위험하지 않은 곳인지 체크
            if danger_map[nx, ny] == 0:
                safe_moves.append(action)

    # --- 행동 결정 로직 ---

    # Case A: 현재 위치가 위험하다! (폭탄 범위 안) -> 무조건 탈출
    if danger_map[x, y] == 1:
        self.logger.debug("DANGER! Escaping...")
        # BFS로 가장 가까운 '안전한 타일(danger_map==0)'을 찾아서 그쪽으로 이동
        best_escape_move = get_move_to_nearest_safe_tile(arena, danger_map, (x, y), others, bomb_xys)
        if best_escape_move:
            return best_escape_move
        # 탈출구가 없으면... 운에 맡기고 아무 데나 간다 (어차피 죽겠지만)
        return np.random.choice(valid_moves) if valid_moves else 'WAIT'

    # Case B: 안전하다. 코인을 찾으러 가자.
    if coins:
        self.logger.debug("Searching for coins...")
        # BFS로 가장 가까운 코인을 찾아서 이동 (단, 위험한 길은 피함)
        best_coin_move = get_move_to_nearest_target(arena, danger_map, (x, y), coins, others, bomb_xys)
        if best_coin_move:
            return best_coin_move

    # Case C: 코인도 없고 안전하다. 그냥 안전하게 배회.
    if safe_moves:
        return safe_moves[0] # 섞여 있으므로 랜덤 선택과 비슷
    
    # Case D: 갈 곳이 없다. (갇힘)
    return 'WAIT'


# --- Helper Functions (BFS) ---

def get_move_to_nearest_safe_tile(arena, danger_map, start, others, bomb_xys):
    """위험지역에서 가장 가까운 안전지대로 가는 첫 번째 행동 반환"""
    queue = deque([(start, None)]) # (위치, 첫 번째 행동)
    visited = set([start])

    while queue:
        (cx, cy), first_action = queue.popleft()
        
        # 안전지대 도착?
        if danger_map[cx, cy] == 0:
            return first_action

        # 4방향 탐색
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            # 이동 가능 조건 (벽X, 타인X, 폭탄X) - 위험지역이어도 통과는 해야 함
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and 
                (nx, ny) not in others and 
                (nx, ny) not in bomb_xys and 
                (nx, ny) not in visited):
                
                visited.add((nx, ny))
                # 첫 행동(first_action)을 유지하며 전파
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None

def get_move_to_nearest_target(arena, danger_map, start, targets, others, bomb_xys):
    """타겟(코인 등)까지의 최단 경로를 찾되, 위험한 길은 피함"""
    queue = deque([(start, None)])
    visited = set([start])

    while queue:
        (cx, cy), first_action = queue.popleft()
        
        if (cx, cy) in targets:
            return first_action

        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        shuffle(directions)
        
        for action, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            
            # 이동 가능 조건 + 위험지역 회피(danger_map[nx, ny] == 0)
            if (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and 
                arena[nx, ny] == 0 and 
                (nx, ny) not in others and 
                (nx, ny) not in bomb_xys and 
                danger_map[nx, ny] == 0 and # 중요: 코인 찾으러 갈 땐 위험지역 밟지 않음
                (nx, ny) not in visited):
                
                visited.add((nx, ny))
                next_action = action if first_action is None else first_action
                queue.append(((nx, ny), next_action))
    return None