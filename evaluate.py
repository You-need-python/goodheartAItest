"""
학습된 Q-table을 불러와, 학습 맵(train)과 일반화 평가용 맵(holdout)
각각에서 완전 탐욕적 정책(ε=0)으로 평가한다.

핵심 설계 원칙: 이 파일은 reward_configs.py를 전혀 import하지 않는다.
오직 grid_env.step()이 반환하는 info(success, collided, steps_taken 등)
만으로 실제 목표 달성도를 측정한다. 즉, 훈련에 쓰인 대리 보상과
완전히 독립적인 잣대로 평가가 이루어진다.
"""

import random

import numpy as np
import pandas as pd

from config import NUM_EVAL_TRIALS, RANDOM_SEED
from grid_env import GridWorldEnv
from maps import TRAIN_MAP, HOLDOUT_MAP, parse_map
from q_learning_agent import QLearningAgent

VERSIONS = ["v1", "v2", "v3"]


def run_evaluation(agent, grid, start, goal, num_trials=NUM_EVAL_TRIALS):
    """
    고정된 정책(ε=0)으로 num_trials회 시행하고, 실제 목표 관련 지표를 집계한다.

    Returns:
        dict: success_rate, avg_steps, avg_collisions, timeout_rate
    """
    env = GridWorldEnv(grid, start, goal)

    successes = 0
    total_steps = 0
    total_collisions = 0
    timeouts = 0

    for _ in range(num_trials):
        state = env.reset()
        collisions = 0
        done = False
        while not done:
            # epsilon=0.0 : 학습된 정책이 실제로 어떻게 행동하는지 탐험 잡음 없이 측정
            action = agent.choose_action(state, epsilon=0.0)
            state, info, done = env.step(action)
            if info["collided"]:
                collisions += 1

        successes += int(info["success"])
        timeouts += int(info["timeout"])
        total_steps += info["steps_taken"]
        total_collisions += collisions

    return {
        "success_rate": successes / num_trials,
        "avg_steps": total_steps / num_trials,
        "avg_collisions": total_collisions / num_trials,
        "timeout_rate": timeouts / num_trials,
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_grid, train_start, train_goal, _ = parse_map(TRAIN_MAP)
    holdout_grid, holdout_start, holdout_goal, _ = parse_map(HOLDOUT_MAP)

    # Q-table은 (row, col) 좌표로 인덱싱되므로, 두 맵의 크기가 다르면
    # 학습된 정책을 그대로 재사용할 수 없다. 이를 실행 시점에 명시적으로 확인한다.
    assert train_grid.shape == holdout_grid.shape, (
        "TRAIN_MAP과 HOLDOUT_MAP의 크기가 달라 동일한 Q-table을 적용할 수 없습니다."
    )

    maps_to_eval = {
        "train": (train_grid, train_start, train_goal),
        "holdout": (holdout_grid, holdout_start, holdout_goal),
    }

    rows = []

    for version in VERSIONS:
        agent = QLearningAgent(grid_shape=train_grid.shape)
        agent.load(f"q_table_{version}.npy")

        for map_type, (grid, start, goal) in maps_to_eval.items():
            metrics = run_evaluation(agent, grid, start, goal)
            rows.append({"version": version, "map_type": map_type, **metrics})
            print(f"[evaluate] {version} / {map_type}: {metrics}")

    df = pd.DataFrame(rows)
    df.to_csv("eval_log.csv", index=False)
    print("[evaluate] eval_log.csv 저장 완료")


if __name__ == "__main__":
    main()