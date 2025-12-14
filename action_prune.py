import numpy as np
from collections import deque
import copy
import settings as s

# ==========================================
# [설정]
# ==========================================
INT_MAX = 9999.0
BOMBING_TEST = 'lookahead'
NO_KICKING = True 
FLAME_LIFE = s.EXPLOSION_TIMER 

class Action:
    Up = 'UP'
    Down = 'DOWN'
    Left = 'LEFT'
    Right = 'RIGHT'
    Bomb = 'BOMB'
    Stop = 'WAIT'

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

def _opposite_direction(direction):
    if direction == Action.Left:  return Action.Right
    if direction == Action.Right: return Action.Left
    if direction == Action.Up:    return Action.Down
    if direction == Action.Down:  return Action.Up
    return None

def _all_directions(exclude_stop=True):
    dirs = [Action.Left, Action.Right, Action.Up, Action.Down]
    return dirs if exclude_stop else dirs + [Action.Stop]

def _manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def _stop_condition(board, pos, explosion_map=None):
    if not position_on_board(board, pos):
        return True
    if position_is_wall(board, pos) or board[pos] == Item.Crate:
        return True
    if explosion_map is not None and position_is_flames(explosion_map, pos):
        return True
    return False

# ==========================================
# [폭발 범위 계산 로직]
# ==========================================
def _all_bomb_real_life(bombs, arena):
    bomb_real_life = {} 
    bomb_map = {}       
    for b in bombs:
        pos = b['position']
        # [중요] 게임 내 타이머가 1이면 이번 턴 직후에 터짐 -> 즉, 이동 중에 터짐
        # 안전하게 하려면 timer를 그대로 씀
        bomb_real_life[pos] = b['timer']
        bomb_map[pos] = b['power']
        
    changed = True
    while changed:
        changed = False
        sorted_bombs = sorted(bomb_real_life.items(), key=lambda x: x[1])
        
        for (bx, by), timer in sorted_bombs:
            power = bomb_map[(bx, by)]
            for d in _all_directions(exclude_stop=True):
                for i in range(1, power + 1):
                    tx, ty = bx, by
                    if d == Action.Up: ty -= i
                    elif d == Action.Down: ty += i
                    elif d == Action.Left: tx -= i
                    elif d == Action.Right: tx += i
                    target = (tx, ty)
                    
                    if not position_on_board(arena, target) or position_is_wall(arena, target):
                        break
                    # 유폭 체크
                    if target in bomb_real_life:
                        if bomb_real_life[target] > timer:
                            bomb_real_life[target] = timer
                            changed = True
    return bomb_real_life, bomb_map

def _position_covered_by_bomb(pos, bomb_real_life, bomb_map, arena):
    min_life = INT_MAX
    max_life = -INT_MAX
    is_covered = False
    
    for (bx, by), timer in bomb_real_life.items():
        power = bomb_map[(bx, by)]
        
        # 1. 같은 행/열인지 체크
        if bx != pos[0] and by != pos[1]: continue
        
        # 2. 거리 체크
        dist = _manhattan_distance((bx, by), pos)
        if dist > power: continue
            
        # 3. 벽에 막히는지 체크
        blocked = False
        if bx == pos[0]: # 수직
            start, end = min(by, pos[1]), max(by, pos[1])
            for y in range(start + 1, end):
                if position_is_wall(arena, (bx, y)): blocked = True; break
        else: # 수평
            start, end = min(bx, pos[0]), max(bx, pos[0])
            for x in range(start + 1, end):
                if position_is_wall(arena, (x, by)): blocked = True; break
                    
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
        
        # 1. 현재 step에서 이미 불길인 곳은 불가
        if explosion_map[curr_pos] > step: continue
        
        is_covered, min_life, max_life = _position_covered_by_bomb(curr_pos, bomb_real_life, bomb_map, arena)
        
        # 안전 지대 발견! (폭발 범위 밖)
        if not is_covered: return step
        
        # 폭발 범위 안인데, 폭발하기 전(min_life)에 도착한 상태라면?
        # -> 그곳에서 폭발을 맞게 되므로 이동 불가 (가지치기)
        if is_covered and step >= min_life: continue
        
        # 폭발은 피했는데(step < min_life), 그 뒤에 불길이 지속되는 동안(max_life + 2) 갇히는지?
        # -> 이 로직은 너무 복잡하므로, 일단 "폭발 전 탈출"만 본다.
            
        for d in _all_directions(exclude_stop=True):
            next_pos = get_next_position(curr_pos, d)
            if position_on_board(arena, next_pos) and \
               not position_is_wall(arena, next_pos) and \
               arena[next_pos] != Item.Crate and \
               next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, step + 1))
    return INT_MAX

# ==========================================
# [Main Function]
# ==========================================
def _compute_safe_actions(obs):
    arena = obs['field']
    explosion_map = obs['explosion_map']
    my_info = obs['self']
    my_pos = my_info[3]
    
    bombs = []
    for b in obs['bombs']:
        bombs.append({'position': b[0], 'timer': b[1], 'power': s.BOMB_POWER})
        
    safe_actions = []
    
    # 1. 물리적 이동 가능성 (벽, 상자)
    possible_moves = []
    for d in _all_directions(exclude_stop=False):
        if d == Action.Stop: next_pos = my_pos
        else: next_pos = get_next_position(my_pos, d)
            
        if not _stop_condition(arena, next_pos): # 벽/상자/현재불길 체크
             possible_moves.append((d, next_pos))

    # 2. 미래 생존성 체크 (폭발 예측)
    bomb_real_life, bomb_map = _all_bomb_real_life(bombs, arena)
    
    for action, next_pos in possible_moves:
        # A. 폭발 직격 체크
        # next_pos로 이동했을 때(1턴 소모), 그 자리가 폭발 범위인가?
        is_covered, min_life, max_life = _position_covered_by_bomb(next_pos, bomb_real_life, bomb_map, arena)
        
        # [수정됨] 타이머가 0이 되면 터짐. 
        # 내가 1턴 써서 이동했는데(t=1), 폭탄이 1턴 뒤에 터지면(min_life=1) -> 사망.
        # min_life가 0이면 이미 터진 것(불길 체크에서 걸러짐).
        if is_covered and min_life <= 1: 
            continue 
        
        # B. 불길 잔존 체크 (explosion_map 활용 강화)
        # 지금은 불길이 없지만(값 0), 내가 도착할 때쯤(1턴 뒤) 이전 폭발의 여파가 남아있나?
        # -> environment.py의 explosion_map 로직에 따르면 timer가 줄어듦.
        # -> explosion_map[next_pos] == 2라면 다음 턴에 1이 되어 여전히 위험.
        if explosion_map[next_pos] > 1:
            continue

        # C. 탈출 가능성 체크 (Doomed Check)
        # next_pos에 도착한 후(step=1), 폭탄이 터지기 전(min_life)에 안전지대로 도망갈 수 있는가?
        # BFS 시작 step을 1로 설정 (이미 이동했으므로)
        min_evade = _compute_min_evade_step(arena, explosion_map, bombs, next_pos, [])
        
        # 내가 그 자리에 도착하는 시간(1) + 탈출 시간(min_evade) >= 폭발 시간(min_life)
        # 즉, 탈출하기도 전에 터진다면 그곳은 사지(Dead End)
        if is_covered and (1 + min_evade) >= min_life:
            continue
            
        safe_actions.append(action)
        
    return safe_actions

def get_filtered_actions(obs, prev_two_obs=None):
    safe_actions = _compute_safe_actions(obs)
    if not safe_actions:
        return [Action.Stop] # 죽음 확정 시 정지
    return safe_actions
