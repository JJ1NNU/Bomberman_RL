import subprocess
import random

# 커리큘럼 단계 정의
CURRICULUM_STAGES = {
    1: "stop_agent",          # 1단계: 가만히 있는 적
    2: "smart_random_no_bomb",  # 2단계: 폭탄 안쓰는 똑똑한 적
    3: "rule_based_agent"   # 3단계: 폭탄 쓰는 똑똑한 적
}

def run_curriculum_episode(my_agent_name, stage_level, episode_id):
    opponent_name = CURRICULUM_STAGES[stage_level]
    
    # 내 팀(0, 1번): 학습 대상 / 적 팀(2, 3번): 커리큘럼 상대
    agents_args = [
        my_agent_name, my_agent_name,  # Team 1 (Us)
        opponent_name, opponent_name   # Team 2 (Opponent)
    ]
    
    cmd = [
        "python", "main.py", "play",
        "--agents", *agents_args,
        "--train", "2",     # 우리 에이전트는 학습 모드
        "--no-gui",
        "--save-replay",    # IRL용 데이터 저장
        "--match-name", f"stage{stage_level}_ep{episode_id}"
    ]
    subprocess.run(cmd)

# 사용 예시
# 1단계 500판 -> 2단계 500판 진행
if __name__ == "__main__":
    for i in range(500):
        run_curriculum_episode("my_rl_agent", 1, i)
    for i in range(500):
        run_curriculum_episode("my_rl_agent", 2, i)
