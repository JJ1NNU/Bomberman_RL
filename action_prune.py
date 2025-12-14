import numpy as np
from collections import deque
import copy
import settings as s
import logging
logger = logging.getLogger('BombeRLeWorld')
# ==========================================
# [설정] 프로젝트 환경에 맞게 상수 재정의
# ==========================================
INT_MAX = 9999.0
BOMBING_TEST = 'lookahead'
NO_KICKING = True 
FLAME_LIFE = s.EXPLOSION_TIMER 

# Action Enum 매핑
class Action:
    Up = 'UP'
    Down = 'DOWN'
    Left = 'LEFT'
    Right = 'RIGHT'
    Bomb = 'BOMB'
    Stop = 'WAIT'

# Item Enum 매핑
class Item:
    Passage = 0
    Crate = 1
    Bomb = 3
    Flames = 4
    Wall = -1

# ==========================================
# [유틸리티]
# ==========================================
def get_next_position(pos, action):
    x, y = pos
    if action == Action.Up:    return (x, y - 1)
    if action == Action.Down:  return (x, y + 1)
    if action == Action.Left:  return (x - 1, y)
    if action == Action.Right: return (x + 1, y)
    return (x, y)

def position_on_board(board, pos):
    x, y = pos
    return 0 <= x < s.COLS and 0 <= y < s.ROWS

def position_is_wall(board, pos):
    return board[pos] == Item.Wall

def position_is_passage(board, pos):
    return board[pos] == Item.Passage

def position_is_flames(explosion_map, pos):
    return explosion_map[pos] > 0

def _manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def _opposite_direction(direction):
    if direction == Action.Left:  return Action.Right
    if direction == Action.Right: return Action.Left
    if direction == Action.Up:    return Action.Down
    if direction == Action.Down:  return Action.Up
    return None

def _all_directions(exclude_stop=True):
    dirs = [Action.Left, Action.Right, Action.Up, Action.Down]
    return dirs if exclude_stop else dirs + [Action.Stop]

def _stop_condition(board, pos, explosion_map=None, current_pos=None):
    if not position_on_board(board, pos):
        return True
    if position_is_wall(board, pos) or board[pos] == Item.Crate:
        return True
    if board[pos] == Item.Bomb:
        if current_pos is not None and pos == current_pos:
            pass
        else:
            return True
    if explosion_map is not None and position_is_flames(explosion_map, pos):
        return True
    return False

# ==========================================
# [팀원 식별 로직]
# ==========================================
def _get_teammate_pos(obs):
    my_name = obs['self'][0]
    teammates_pos = []
    
    # 1. 학습 환경 (environment.py에서 Team 접두사 부여)
    if my_name.startswith("Team"):
        my_team_tag = my_name.split('_')[0] # "Team1"
        for other in obs['others']:
            if other[0].startswith(my_team_tag):
                teammates_pos.append(other[3])
                
    # 2. 대회/일반 환경 (이름 유사도 기반)
    else:
        my_base = my_name.rsplit('_', 1)[0] if '_' in my_name else my_name
        for other in obs['others']:
            other_base = other[0].rsplit('_', 1)[0] if '_' in other[0] else other[0]
            if my_base == other_base:
                teammates_pos.append(other[3])
                
    return teammates_pos

# ==========================================
# [폭탄 시뮬레이션 로직]
# ==========================================
def _all_bomb_real_life(bombs, arena):
    bomb_real_life = {}
    bomb_map = {}
    
    for b in bombs:
        pos = b['position']
        # 유폭이 없으므로, 폭탄의 '실제 폭발 시간'은 '현재 타이머'와 동일함.
        bomb_real_life[pos] = b['timer']
        bomb_map[pos] = b['power']
        
    return bomb_real_life, bomb_map

def _position_covered_by_bomb(pos, bomb_real_life, bomb_map, arena):
    """
    특정 위치(pos)가 폭발 범위에 들어가는지 판단.
    * 규칙 1: 상자(Crate)는 폭발을 막지 못하고 투과됨.
    * 규칙 2: 벽(Wall)만 폭발을 막음.
    """
    min_life = INT_MAX
    max_life = -INT_MAX
    is_covered = False
    
    px, py = pos

    for (bx, by), timer in bomb_real_life.items():
        power = bomb_map[(bx, by)]
        
        # 1. 같은 행/열인지 체크
        if bx != px and by != py: 
            continue
        
        # 2. 거리 체크
        dist = abs(bx - px) + abs(by - py)
        if dist > power: 
            continue

        # 3. 벽(Wall) 체크 (상자는 투과됨!)
        blocked = False
        
        if bx == px: # 수직선
            step = 1 if by < py else -1
            for y in range(py - step, by, -step):
                if position_is_wall(arena, (bx, y)): # Wall만 체크
                    blocked = True
                    break
        else: # 수평선
            step = 1 if bx < px else -1
            for x in range(px - step, bx, -step):
                if position_is_wall(arena, (x, by)): # Wall만 체크
                    blocked = True
                    break
        
        if not blocked:
            is_covered = True
            min_life = min(min_life, timer)
            max_life = max(max_life, timer)
            
    return is_covered, min_life, max_life



def _compute_min_evade_step(arena, explosion_map, bombs, my_pos, history_pos):
    bomb_real_life, bomb_map = _all_bomb_real_life(bombs, arena)
    queue = deque([(my_pos, 0)])
    visited = set([my_pos])
    
    while queue:
        curr_pos, step = queue.popleft()
        if explosion_map[curr_pos] > step: continue
        is_covered, min_life, max_life = _position_covered_by_bomb(curr_pos, bomb_real_life, bomb_map, arena)
        
        if not is_covered: return step
        if is_covered and step >= min_life: continue
            
        for d in _all_directions(exclude_stop=True):
            next_pos = get_next_position(curr_pos, d)
            if position_on_board(arena, next_pos) and \
               not position_is_wall(arena, next_pos) and \
               arena[next_pos] != Item.Crate and \
               arena[next_pos] != Item.Bomb and \
               next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, step + 1))
    return INT_MAX

# ==========================================
# [Main Filter Logic]
# ==========================================
def _compute_safe_actions(obs):
    arena = obs['field']
    explosion_map = obs['explosion_map']
    
    # [수정] 정확한 좌표 인덱싱 (4번째 요소)
    my_info = obs['self']
    my_pos = my_info[3] 
    
    bombs = []
    for b in obs['bombs']:
        bombs.append({'position': b[0], 'timer': b[1], 'power': s.BOMB_POWER})
        
    safe_actions = []
    
    # [디버그용] 폭탄 정보 요약
    if bombs:
        bomb_info = [f"B{b['position']} T:{b['timer']}" for b in bombs]
        logger.debug(f"💣 Active Bombs: {bomb_info}") 

    # -----------------------------------------------
    # 1. 이동(Move) 행동 필터링 (규칙 1, 2)
    # -----------------------------------------------
    possible_moves = []
    for d in _all_directions(exclude_stop=False):
        if d == Action.Stop: next_pos = my_pos
        else: next_pos = get_next_position(my_pos, d)
            
        # 기본 물리적 충돌 체크 (벽, 상자, 불길)
        if not _stop_condition(arena, next_pos, explosion_map, current_pos=my_pos): # explosion_map 넘겨서 불길 체크
             possible_moves.append((d, next_pos))

    bomb_real_life, bomb_map = _all_bomb_real_life(bombs, arena)
    
    for action, next_pos in possible_moves:
        is_covered, min_life, max_life = _position_covered_by_bomb(next_pos, bomb_real_life, bomb_map, arena)
        
        # 에이전트의 판단 과정 로그 찍기
        # 예: "RIGHT로 가면 (1,2)인데, 거긴 폭발 예상됨(Covered). 남은시간(Life): 2초"
        if is_covered:
            # 위험한 경우만 로그로 확인 (너무 많으니까)
            logger.debug(f"Action {action} to {next_pos} is DANGEROUS! Life: {min_life}")
            pass

        # A. 완전히 안전한 곳이면 OK
        if not is_covered:
            safe_actions.append(action)
            continue
            
        # B. 폭발 범위 내라도, 당장 죽지 않으면(>1) OK
        # 이동에 대해서는 미래의 탈출 가능성(min_evade)을 따지지 않음.
        # 일단 움직이고 나서 생각하게 함.
        if min_life > 1:
            if action == Action.Stop:
                continue
            safe_actions.append(action)
        
    # -----------------------------------------------
    # 2. 폭탄(Bomb) 행동 필터링 (규칙 3, 4, 5)
    # -----------------------------------------------
    can_bomb = my_info[2] # bombs_left
    
    if can_bomb:
        # [규칙 3] 현재 위치가 이미 폭발 범위 안이면 금지
        is_covered, _, _ = _position_covered_by_bomb(my_pos, bomb_real_life, bomb_map, arena)
        
        if not is_covered:
            # [규칙 4] 팀원 보호: 팀원이 맞을 위치에 있으면 폭탄 금지
            teammates = _get_teammate_pos(obs)
            is_teammate_close = False
            for t_pos in teammates:
                virtual_bomb_map = {my_pos: s.BOMB_POWER}
                virtual_bomb_real_life = {my_pos: 0} # 즉시 폭발 가정
                is_hit, _, _ = _position_covered_by_bomb(t_pos, virtual_bomb_real_life, virtual_bomb_map, arena)
                
                if is_hit:
                    is_teammate_close = True
                    break
            
            if not is_teammate_close:
                # [규칙 5] 자폭 방지(Lookahead): 폭탄 놓고 10틱 안에 탈출 가능한가?
                virtual_bomb = {'position': my_pos, 'timer': 10, 'power': s.BOMB_POWER}
                simulated_bombs = bombs + [virtual_bomb]
                
                # 가상 환경에서 생존 가능성 체크
                min_evade = _compute_min_evade_step(arena, explosion_map, simulated_bombs, my_pos, [])
                
                # 폭탄 설치는 이동보다 신중해야 하므로 완화 조건(min_life > 2)을 적용하지 않음.
                # 확실히 도망갈 수 있을 때만 설치.
                if min_evade < 10: 
                    safe_actions.append(Action.Bomb)
                
                # [★ 추가된 로직] 상자를 부수기 위한 공격적 설치 허용 (Aggressive Bombing)
                # min_evade가 실패했더라도(상자에 막힘), 
                # 현재 내 위치에서 폭발 범위 밖으로 나갈 수 있는 '빈 공간'이 충분하다면 설치 허용.
                else:
                    # BFS로 '빈 공간(Passage)'의 깊이(Depth) 탐색
                    # 내 위치에서 상자/벽 없이 갈 수 있는 칸이 4칸 이상이면 도망갈 수 있다고 간주.
                    safe_space_depth = _measure_safe_space_depth(arena, my_pos, s.BOMB_POWER + 1)
                    
                    if safe_space_depth > s.BOMB_POWER: 
                        # 도망갈 구멍이 충분하므로 설치 허용! (상자는 터질 거니까 걱정 마)
                        safe_actions.append(Action.Bomb)

    return safe_actions

def get_filtered_actions(obs, prev_two_obs=None):
    safe_actions = _compute_safe_actions(obs)
    
    # 정말 갈 곳이 없으면 Stop이라도 반환 (에러 방지)
    if not safe_actions:
        return [Action.Stop]
        
    return safe_actions

def _measure_safe_space_depth(arena, start_pos, max_depth):
    """
    현재 위치에서 상자나 벽을 만나지 않고 이동할 수 있는 최대 거리(Depth)를 측정.
    (폭탄 놓고 튈 공간이 있는지 확인용)
    """
    queue = deque([(start_pos, 0)])
    visited = set([start_pos])
    max_d = 0
    
    while queue:
        curr, depth = queue.popleft()
        max_d = max(max_d, depth)
        
        if depth >= max_depth:
            return depth
            
        for d in _all_directions(exclude_stop=True):
            next_pos = get_next_position(curr, d)
            
            if position_on_board(arena, next_pos) and \
               not position_is_wall(arena, next_pos) and \
               arena[next_pos] != Item.Crate and \
               next_pos not in visited:
                
                visited.add(next_pos)
                queue.append((next_pos, depth + 1))
                
    return max_d
