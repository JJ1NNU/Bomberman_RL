import numpy as np
import settings as s
import logging
from collections import deque

logger = logging.getLogger('BombeRLeWorld')

# ==========================================
# [설정]
# ==========================================
# 팀원과 유지할 최소 맨해튼 거리 (이 거리보다 가까워지면 이동 제한)
# 1이면 바로 옆칸 허용 안 함, 0이면 겹치는 것만 방지.
# 좁은 맵 특성상 작은 수 설정하여 '바로 옆'에 붙는 것을 방지하는 것이 좋습니다.
MIN_TEAM_DIST = 2

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

def position_is_flames(explosion_map, pos):
    return explosion_map[pos] > 0

def _manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def _all_directions(exclude_stop=True):
    dirs = [Action.Left, Action.Right, Action.Up, Action.Down]
    return dirs if exclude_stop else dirs + [Action.Stop]

def _stop_condition(board, pos, explosion_map=None, current_pos=None):
    # 1. 맵 밖 체크
    if not position_on_board(board, pos):
        return True
    
    # 2. 물리적 장애물(벽, 상자) 체크
    if position_is_wall(board, pos) or board[pos] == Item.Crate:
        return True
    
    # 3. 폭탄 체크 (이미 놓인 폭탄 위로는 못 감)
    if board[pos] == Item.Bomb:
        if current_pos is not None and pos == current_pos:
            pass
        else:
            return True
            
    # 4. 불길(Flames) 체크: 불 속으로는 안 들어감
    if explosion_map is not None and position_is_flames(explosion_map, pos):
        return True
        
    return False

# ==========================================
# [Loop 감지 로직]
# ==========================================
def _is_looping_action(action, history):
    """
    최근 행동 패턴이 반복되는지 확인.
    패턴: A -> B -> A -> B 상태에서 다시 A를 하려고 하면 True 반환.
    history: 최근 행동들이 담긴 리스트 또는 데크 (문자열)
    """
    if history is None or len(history) < 4:
        return False
        
    # history[-1]은 가장 최근 행동
    # 예: history = [..., 'UP', 'DOWN', 'UP', 'DOWN']
    # next_action = 'UP'
    
    # 패턴 확인: (전전 == 전전전전) AND (전 == 전전전)
    # A(t-4) == A(t-2) AND B(t-3) == B(t-1)
    if history[-2] == history[-4] and history[-1] == history[-3]:
        # 이번 행동이 A(t-2)와 같다면 루프 시도임
        if action == history[-2]:
            return True
            
    return False

# ==========================================
# [팀원 식별 로직]
# ==========================================
def _get_teammate_pos(obs):
    my_name = obs['self'][0]
    teammates_pos = []
    
    if my_name.startswith("Team"):
        my_team_tag = my_name.split('_')[0] 
        for other in obs['others']:
            if other[0].startswith(my_team_tag):
                teammates_pos.append(other[3])
    else:
        my_base = my_name.rsplit('_', 1)[0] if '_' in my_name else my_name
        for other in obs['others']:
            other_base = other[0].rsplit('_', 1)[0] if '_' in other[0] else other[0]
            if my_base == other_base:
                teammates_pos.append(other[3])
                
    return teammates_pos

# ==========================================
# [폭탄 시뮬레이션 로직 - 팀원 피해 계산용]
# ==========================================
def _position_covered_by_bomb(pos, bomb_pos, power, arena):
    px, py = pos
    bx, by = bomb_pos
    if bx != px and by != py: return False
    dist = abs(bx - px) + abs(by - py)
    if dist > power: return False
    blocked = False
    if bx == px: 
        step = 1 if by < py else -1
        for y in range(py - step, by, -step):
            if position_is_wall(arena, (bx, y)): 
                blocked = True
                break
    else:
        step = 1 if bx < px else -1
        for x in range(px - step, bx, -step):
            if position_is_wall(arena, (x, by)): 
                blocked = True
                break
    return not blocked

# ==========================================
# [Main Filter Logic]
# ==========================================
def _compute_team_safe_actions(obs, action_history, ignore_team_dist=False, ignore_loop=False):
    """
    ignore_team_dist: True일 경우 팀원 거리 유지 제약을 무시함.
    ignore_loop: True일 경우 루프 방지 제약을 무시함.
    """
    arena = obs['field']
    explosion_map = obs['explosion_map']
    my_info = obs['self']
    my_pos = my_info[3] 
    teammates = _get_teammate_pos(obs)

    safe_actions = []

    # -----------------------------------------------
    # 1. 이동(Move) 행동 필터링
    # -----------------------------------------------
    for d in _all_directions(exclude_stop=False):
        if d == Action.Stop: next_pos = my_pos
        else: next_pos = get_next_position(my_pos, d)
            
        # A. 기본 물리적/불길 안전 체크 (이건 절대 양보 불가)
        if _stop_condition(arena, next_pos, explosion_map, current_pos=my_pos):
            continue

        # B. 팀원 거리 유지 체크
        if not ignore_team_dist:
            too_close = False
            for t_pos in teammates:
                if _manhattan_distance(next_pos, t_pos) <= MIN_TEAM_DIST:
                    too_close = True
                    break
            if too_close:
                continue

        # C. Loop 방지 체크
        if not ignore_loop:
            if d != Action.Stop and _is_looping_action(d, action_history):
                continue

        safe_actions.append(d)

    # -----------------------------------------------
    # 2. 폭탄(Bomb) 행동 필터링 (팀킬 방지는 항상 적용)
    # -----------------------------------------------
    can_bomb = my_info[2]
    if can_bomb:
        is_teammate_danger = False
        for t_pos in teammates:
            if _position_covered_by_bomb(t_pos, my_pos, s.BOMB_POWER, arena):
                is_teammate_danger = True
                break
        
        if not is_teammate_danger:
            safe_actions.append(Action.Bomb)

    return safe_actions


def get_filtered_actions(obs, action_history=None):
    """
    obs: 게임 상태
    action_history: 에이전트의 과거 행동 리스트
    """
    # 1차 시도: [Strict] 팀원 거리 유지 O + Loop 방지 O
    safe_actions = _compute_team_safe_actions(obs, action_history, ignore_team_dist=False, ignore_loop=False)
    
    # 2차 시도: [Relaxed Dist] 팀원 거리 유지 X + Loop 방지 O
    if not safe_actions:
        safe_actions = _compute_team_safe_actions(obs, action_history, ignore_team_dist=True, ignore_loop=False)

    # 3차 시도: [Relaxed All] 팀원 거리 유지 X + Loop 방지 X
    # -> 에이전트가 루프를 돌더라도 갈 수 있는 곳은 다 열어줌 (WAIT 강제 방지)
    if not safe_actions:
        # logger.debug("All filters failed. Allowing loops/crowding to prevent stuck.")
        safe_actions = _compute_team_safe_actions(obs, action_history, ignore_team_dist=True, ignore_loop=True)

    # 4차: 그래도 없다면(물리적으로 갇힘) 어쩔 수 없이 WAIT
    if not safe_actions:
        return [Action.Stop]
        
    return safe_actions