import subprocess
import random
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# [설정] 에이전트 풀 정의: 추가 학습이 가능한 에이전트들
# ==========================================
AGENT_POOL = [
    "expert_2",
    "hung_ry_agent"
]

# 동시에 실행할 프로세스 수 (CPU 코어 수에 맞춰 조절, 보통 4~8)
MAX_WORKERS = 4

def run_match(team1_agents, team2_agents, match_id, training_mode=False, save_replay=True):
    """
    개별 매치를 실행하는 함수
    """
    agents_args = [
        team1_agents[0], 
        team1_agents[1], 
        team2_agents[0], 
        team2_agents[1]  
    ]
    
    # 0: Eval, 2: Train (Team 1 only), 4: Train (All)
    train_arg = "2" if training_mode else "0"

    # main.py 실행 커맨드
    cmd = [
        "python", "main.py", "play",
        "--agents", *agents_args,
        "--n-rounds", "100",            
        "--train", train_arg,          
        "--no-gui",                    
        "--match-name", f"league_{match_id}", 
        "--save-stats"                 
    ]
    
    # [중요] 데이터 수집용 리플레이 저장
    if save_replay:
        cmd.append("--save-replay")
    
    start_time = datetime.now().strftime('%H:%M:%S')
    print(f"[{start_time}] Match {match_id} STARTED: {team1_agents} vs {team2_agents}")
    
    try:
        # capture_output=True를 쓰면 로그가 너무 길어질 수 있으니 필요시 조정
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Match {match_id} FINISHED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in match {match_id}: {e}")
        print(e.stderr)
        return False

def run_league(num_matches=50, save_replay=True):
    print(f"Starting League with {num_matches} matches...")
    print(f"Parallel Workers: {MAX_WORKERS}")
    
    # 매치 큐 생성
    match_tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(num_matches):
            # 1. 에이전트 랜덤 선택 (중복 허용 여부에 따라 sample vs choices 결정)
            # 여기서는 풀에서 서로 다른 2명을 뽑아 Team 1 구성
            selected = random.sample(AGENT_POOL, k=2)
            
            team1 = [selected[0], selected[1]]
            # Team 2는 베이스라인(Rule Based)으로 고정하여 실력 검증
            team2 = ["rule_based_agent", "rule_based_agent"]
            
            # 2. 작업 예약
            future = executor.submit(
                run_match, 
                team1_agents=team1, 
                team2_agents=team2, 
                match_id=i, 
                training_mode=True,  # 평가데이터는 league_runner_2v2.py에서 모으자. 여긴 학습용.
                save_replay=save_replay
            )
            match_tasks.append(future)

        # 3. 완료 대기 및 결과 확인
        completed_count = 0
        for future in as_completed(match_tasks):
            if future.result():
                completed_count += 1
                
    print(f"League Finished. Successfully completed matches: {completed_count}/{num_matches}")

if __name__ == "__main__":
    if not os.path.exists("agent_code"):
        print("Error: 'agent_code' directory not found.")
        exit(1)
        
    # 리플레이 저장이 필요하면 True로 설정
    run_league(num_matches=1000, save_replay=False)
