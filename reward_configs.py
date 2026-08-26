"""
대리 보상(proxy reward) 함수 정의.

각 함수는 grid_env.GridWorldEnv.step() 이 반환하는 info 딕셔너리만을
입력으로 받아 하나의 실수 보상을 계산한다. 함수는 환경이나 에이전트의
내부 상태에 직접 접근하지 않는다.

이렇게 설계해야 "보상 함수를 바꾸는 것"이 환경 코드(grid_env.py)나
평가 코드(evaluate.py)의 수정 없이, REWARD_REGISTRY 에서 이름만
바꿔 조회하는 것으로 이루어질 수 있다.
"""

STEP_PENALTY = -0.05       # 매 스텝마다 부과되는 기본 페널티 (이동 효율을 유도, V1~V3 공통)
COLLISION_PENALTY = -1.0   # V2, V3 에서 충돌 시 추가로 부과되는 페널티
REVISIT_PENALTY = -0.2     # V3 에서 이미 방문한 칸에 재도달했을 때 추가로 부과되는 페널티
GOAL_REWARD = 10.0         # 목적지 도달 시 공통으로 지급되는 보상


def _distance_shaping(info):
    """목적지와의 거리가 줄었으면 양수, 늘었으면 음수가 되는 shaping 보상."""
    return info["prev_distance"] - info["curr_distance"]


def reward_v1(info):
    """
    버전 1: 목적지에 가까워지는 정도만 반영한다.
    장애물과의 충돌이나 비효율적인 이동(재방문)에는 어떠한 페널티도 없다.
    """
    if info["success"]:
        return GOAL_REWARD

    return _distance_shaping(info) + STEP_PENALTY


def reward_v2(info):
    """
    버전 2: V1 의 보상에 충돌 페널티를 추가한다.
    """
    reward = reward_v1(info)

    if info["collided"]:
        reward += COLLISION_PENALTY

    return reward


def reward_v3(info):
    """
    버전 3: V2 의 보상에 재방문(비효율적 이동) 페널티를 추가한다.
    """
    reward = reward_v2(info)

    if info["revisited"]:
        reward += REVISIT_PENALTY

    return reward


REWARD_REGISTRY = {
    "v1": reward_v1,
    "v2": reward_v2,
    "v3": reward_v3,
}