import subprocess
import random
import os
from datetime import datetime

# ==========================================
# [설정] 에이전트 풀 정의
# agent_codes 폴더 내에 존재하는 폴더 이름들이어야 합니다.
# ==========================================
AGENT_POOL = [
    "agent_the_destroyer_of_universes"
    "expert_1"
    "expert_2"
    "coin_collector_agent"
    "hung_ry_agent"
    "rule_based_agent"
    "the_second_agent_to_rule_them_all"
    "agent_fred2"
    "feature_is_everything"
]

def run_match(team1_agents, team2_agents, match_id, training_mode=False):
    """
    Args:
        team1_agents (list): Team 1을 구성할 에이전트 이름 2개 [A1, A2]
        team2_agents (list): Team 2를 구성할 에이전트 이름 2개 [B1, B2]
        match_id (int): 매치 식별 번호
        training_mode (bool): True일 경우 --train 인자를 사용하여 학습 모드 활성화
    """
    
    #  0, 1번 인덱스가 Team 1 / 2, 3번 인덱스가 Team 2
    agents_args = [
        team1_agents[0], # Agent 0 (Team 1)
        team1_agents[1], # Agent 1 (Team 1)
        team2_agents[0], # Agent 2 (Team 2)
        team2_agents[1]  # Agent 3 (Team 2)
    ]
    
    # --train 옵션 설정
    # 리그전 데이터 수집용이라면 보통 0 (모두 추론 모드)으로 설정합니다.
    # 만약 우리 팀 에이전트를 학습시키면서 돌리는 경우라면 2로 설정합니다.
    train_arg = "2" if training_mode else "0"

    # main.py 실행 커맨드 구성
    cmd = [
        "python", "main.py", "play",
        "--agents", *agents_args,
        "--n-rounds", "10",            # 매치 당 라운드 수 (필요에 따라 조절)
        "--train", train_arg,          # 0: 모든 에이전트 Evaluation 모드
        "--no-gui",                    # 빠른 데이터 수집을 위해 GUI 끄기
        "--save-replay",               # ★ 중요: 전문가 데이터 추출을 위한 리플레이 저장
        "--match-name", f"league_{match_id}", # 로그 파일 식별자
        "--save-stats"                 # 승률 분석을 위한 통계 저장
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Match {match_id} Start: {team1_agents} vs {team2_agents}")
    
    # 서브프로세스 실행 (에러 발생 시 무시하고 다음 매치로 진행하려면 try-except 사용)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in match {match_id}: {e}")

def run_league(num_matches=100):
    """
    랜덤 매칭 리그를 진행합니다.
    """
    print(f"Starting League with {num_matches} matches...")
    print(f"Agent Pool: {AGENT_POOL}")
    
    for i in range(num_matches):
        # 1. 에이전트 풀에서 랜덤하게 4개 선택 (중복 허용: 동일 에이전트 간 대결 가능)
        # 만약 서로 다른 종류만 붙이고 싶다면 random.sample을 사용하세요.
        selected = random.choices(AGENT_POOL, k=4)
        
        # 2. 팀 구성 (순서대로 0,1번이 한 팀 / 2,3번이 한 팀)
        team1 = [selected[0], selected[1]]
        team2 = [selected[2], selected[3]]
        
        # 3. 매치 실행
        run_match(team1, team2, match_id=i)

if __name__ == "__main__":
    # 실행 전 agent_codes 폴더 확인
    if not os.path.exists("agent_codes"):
        print("Error: 'agent_codes' directory not found.")
        exit(1)
        
    run_league(num_matches=50)
