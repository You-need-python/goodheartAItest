"""
학습 및 평가에 사용되는 모든 하이퍼파라미터를 한 곳에서 관리한다.

V1/V2/V3 세 버전은 이 파일의 값을 동일하게 공유해서 사용해야 한다.
그래야 "보상 함수 외의 모든 조건은 동일하다"는 통제된 실험 조건이
코드 수준에서도 보장된다.
"""

# ---- Q-learning 하이퍼파라미터 ----
ALPHA = 0.1          # 학습률
GAMMA = 0.95         # 할인율

# ---- ε-greedy 탐험 스케줄 ----
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 800   # 이 에피소드 수에 걸쳐 EPSILON_START -> EPSILON_END 로 선형 감소

# ---- 훈련/평가 규모 ----
NUM_TRAIN_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100
NUM_EVAL_TRIALS = 200

# ---- 재현성 ----
RANDOM_SEED = 42

# ---- 행동 정의 (4방향, 대각선 없음) ----
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

NUM_ACTIONS = len(ACTIONS)