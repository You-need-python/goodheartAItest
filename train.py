"""
V1/V2/V3 각각에 대해 동일한 맵·하이퍼파라미터·난수 시드로 Q-learning
훈련을 수행하고, 에피소드별 로그를 train_log.csv에 기록한다.

주의: 이 파일에서만 reward_configs.REWARD_REGISTRY를 참조하여 info로부터
proxy_reward를 계산한다. grid_env.py는 이 계산에 관여하지 않는다.
"""

import random

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    NUM_TRAIN_EPISODES,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_EPISODES,
    ALPHA,
    GAMMA,
)
from grid_env import GridWorldEnv
from maps import TRAIN_MAP, parse_map
from q_learning_agent import QLearningAgent
from reward_configs import REWARD_REGISTRY


def epsilon_at(episode):
    """episode 진행에 따라 EPSILON_START에서 EPSILON_END로 선형 감소하는 ε을 계산한다."""
    if episode >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    fraction = episode / EPSILON_DECAY_EPISODES
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def run_training(version, grid, start, goal, num_episodes=NUM_TRAIN_EPISODES):
    """
    지정된 보상 버전으로 한 번의 전체 훈련을 수행한다.

    Args:
        version: "v1" / "v2" / "v3" (REWARD_REGISTRY의 키)
        grid, start, goal: maps.parse_map()의 반환값
        num_episodes: 훈련 에피소드 수

    Returns:
        agent: 학습이 끝난 QLearningAgent
        log_rows: 에피소드별 기록(dict)의 리스트
    """
    reward_fn = REWARD_REGISTRY[version]
    env = GridWorldEnv(grid, start, goal)
    agent = QLearningAgent(grid_shape=grid.shape, alpha=ALPHA, gamma=GAMMA)

    log_rows = []

    for episode in range(num_episodes):
        state = env.reset()
        epsilon = epsilon_at(episode)

        cumulative_proxy_reward = 0.0
        collisions = 0

        done = False
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, info, done = env.step(action)

            # info를 이 버전에 해당하는 보상 함수에 통과시켜 proxy_reward를 계산한다.
            # (환경은 이 계산에 관여하지 않는다.)
            reward = reward_fn(info)
            agent.update(state, action, reward, next_state, done)

            cumulative_proxy_reward += reward
            if info["collided"]:
                collisions += 1

            state = next_state

        log_rows.append({
            "version": version,
            "episode": episode,
            "epsilon": round(epsilon, 4),
            "cumulative_proxy_reward": round(cumulative_proxy_reward, 3),
            "steps": info["steps_taken"],
            "success": info["success"],
            "collisions": collisions,
        })

    return agent, log_rows


def main():
    grid, start, goal, shape = parse_map(TRAIN_MAP)

    all_log_rows = []

    for version in REWARD_REGISTRY:
        # 버전마다 난수 시드를 동일하게 재설정한다. 에피소드 진행에 따라
        # 소비되는 난수의 양이 버전별로 달라질 수 있으므로, 이렇게 재설정해야
        # 세 버전이 정확히 동일한 난수 흐름에서 출발한다는 것을 보장할 수 있다.
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print(f"[train] {version} 훈련 시작...")
        agent, log_rows = run_training(version, grid, start, goal)
        all_log_rows.extend(log_rows)

        agent.save(f"q_table_{version}.npy")
        print(f"[train] {version} 훈련 완료, q_table_{version}.npy 저장")

    df = pd.DataFrame(all_log_rows)
    df.to_csv("train_log.csv", index=False)
    print("[train] train_log.csv 저장 완료")


if __name__ == "__main__":
    main()