import numpy as np
import settings as s

def state_to_features(game_state: dict) -> np.array:
    """
    게임 상태를 7개의 17x17 채널(이미지 형태)로 변환합니다.
    Output Shape: (7, 17, 17)
    """
    if game_state is None:
        return None

    # 0. 초기화
    channels = []
    
    # 1. Walls (벽): 고정된 장애물
    walls = (game_state['field'] == -1).astype(np.float32)
    channels.append(walls)

    # 2. Crates (상자): 파괴 가능한 장애물
    crates = (game_state['field'] == 1).astype(np.float32)
    channels.append(crates)

    # 3. Coins (코인): 먹어야 할 목표
    coins_map = np.zeros((s.COLS, s.ROWS), dtype=np.float32)
    for (xc, yc) in game_state['coins']:
        coins_map[xc, yc] = 1.0
    channels.append(coins_map)

    # 4. Bombs (폭탄): 위험물 (폭발 남은 시간 정규화)
    # 폭탄이 있는 위치에 (timer / 30) 값을 넣음. 없으면 0.
    bombs_map = np.zeros((s.COLS, s.ROWS), dtype=np.float32)
    for (xb, yb), t in game_state['bombs']:
        bombs_map[xb, yb] = t / s.BOMB_TIMER
    channels.append(bombs_map)

    # 5. Explosions (폭발): 즉사 구역 (남은 시간 정규화)
    # explosion_map에는 남은 턴 수가 들어있음
    explosions_map = game_state['explosion_map'] / s.EXPLOSION_TIMER
    channels.append(explosions_map)

    # 6. Others (적): 상대방 위치
    others_map = np.zeros((s.COLS, s.ROWS), dtype=np.float32)
    for _, _, _, (xo, yo) in game_state['others']:
        others_map[xo, yo] = 1.0
    channels.append(others_map)

    # 7. Self (나): 내 위치
    myself_map = np.zeros((s.COLS, s.ROWS), dtype=np.float32)
    x, y = game_state['self'][3]
    myself_map[x, y] = 1.0
    channels.append(myself_map)

    # (7, 17, 17) 형태로 합치기
    observation = np.stack(channels)
    
    return observation
