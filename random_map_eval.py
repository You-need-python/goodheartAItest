"""
무작위 맵 일반화 검증 실험.

저장된 Q-table(q_table_v1/v2/v3.npy)을, 무작위로 생성한 8x8 맵에 적용해
다음 두 부류의 맵에서 성공률을 비교한다.

    - 단조 경로 존재 맵 : DOWN/RIGHT 이동만으로 시작점에서 목적지에 도달 가능
    - 우회 필요 맵      : 일반적으로는 도달 가능(BFS)하지만, DOWN/RIGHT만으로는
                          도달 불가능하여 최소 한 번의 역행(UP/LEFT)이 필요

이 프로젝트의 모든 맵은 시작이 좌상단, 목적지가 우하단으로 고정되어 있으므로,
DOWN/RIGHT는 항상 목적지와의 거리를 줄이는 방향이다. 따라서 "단조 경로 존재
여부"는 목적지에 가까워지는 방향으로만 가는 정책이 풀 수 있는 맵인지를 가르는
기준이 된다.

실험 설계:
    - 벽 확률(WALL_PROBS)마다, 각 부류의 맵을 MAPS_PER_CLASS개씩 수집한다.
    - 세 버전(v1~v3)은 정확히 같은 맵 집합으로 평가한다 (비교의 공정성).
    - 평가 정책은 완전 탐욕적(ε=0)이며, 보상 함수는 평가에 관여하지 않는다.
    - 결과는 random_map_eval.csv 로 저장하고 표로 출력한다.

주의:
    - 이 실험은 지정된 Q-table 한 벌(학습 시드 하나)에 대한 측정이다.
      학습 시드를 바꿔 반복한 결과가 아니다.
    - 벽 확률이 낮을수록 "우회 필요 맵"이 드물어 수집에 시도 횟수가 많이 든다.

실행: python random_map_eval.py
"""

import random

import numpy as np
import pandas as pd

from grid_env import GridWorldEnv
from maps import is_reachable
from q_learning_agent import QLearningAgent

# ---- 실험 설정 ----
GRID_SIZE = 8
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
WALL_PROBS = [0.15, 0.25, 0.35]
MAPS_PER_CLASS = 400
MAX_TRIES = 200_000      # 한 벽 확률에서 맵 생성을 시도하는 최대 횟수 (무한 루프 방지)
MAP_SEED = 2026
VERSIONS = ["v1", "v2", "v3"]

OPTIMAL_STEPS = abs(GOAL[0] - START[0]) + abs(GOAL[1] - START[1])  # 맨해튼 거리

CLASS_LABELS = {
    "monotone": "단조 경로 존재",
    "non_monotone": "우회 필요",
}


def generate_random_grid(rng, wall_prob):
    """시작/목적지 칸을 제외한 모든 칸을 wall_prob 확률로 벽으로 만든다."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) in (START, GOAL):
                continue
            grid[r, c] = rng.random() < wall_prob
    return grid


def has_monotone_path(grid):
    """
    DOWN/RIGHT 이동만으로 START에서 GOAL에 도달할 수 있는지 동적계획법으로 확인한다.
    reach[r, c] = (위 칸에서 올 수 있음) 또는 (왼쪽 칸에서 올 수 있음), 단 벽이 아닌 칸만.
    """
    reach = np.zeros_like(grid, dtype=bool)
    reach[START] = True
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] or (r, c) == START:
                continue
            from_above = r > 0 and reach[r - 1, c]
            from_left = c > 0 and reach[r, c - 1]
            reach[r, c] = from_above or from_left
    return bool(reach[GOAL])


def sample_maps(rng, wall_prob):
    """
    도달 가능한 무작위 맵을 생성해 두 부류로 분류하여 각각 MAPS_PER_CLASS개 수집한다.

    Returns:
        maps: {"monotone": [grid, ...], "non_monotone": [grid, ...]}
        tries: 사용한 생성 시도 횟수
    """
    maps = {"monotone": [], "non_monotone": []}
    tries = 0

    while tries < MAX_TRIES and min(len(v) for v in maps.values()) < MAPS_PER_CLASS:
        tries += 1
        grid = generate_random_grid(rng, wall_prob)

        if not is_reachable(grid, START, GOAL):
            continue

        key = "monotone" if has_monotone_path(grid) else "non_monotone"
        if len(maps[key]) < MAPS_PER_CLASS:
            maps[key].append(grid)

    return maps, tries


def run_episode(agent, grid):
    """완전 탐욕 정책으로 한 에피소드를 실행하고 실제 목표 지표를 반환한다."""
    env = GridWorldEnv(grid, START, GOAL)
    state = env.reset()
    collisions = 0
    done = False

    while not done:
        action = agent.choose_action(state, epsilon=0.0)
        state, info, done = env.step(action)
        if info["collided"]:
            collisions += 1

    return {
        "success": bool(info["success"]),
        "timeout": bool(info["timeout"]),
        "steps": info["steps_taken"],
        "collisions": collisions,
    }


def load_agents():
    agents = {}
    for version in VERSIONS:
        agent = QLearningAgent(grid_shape=(GRID_SIZE, GRID_SIZE))
        agent.load(f"구진로/goodheartAItest/q_table_{version}.npy")
        agents[version] = agent
    return agents


def main():
    # 맵 생성용 난수와 에이전트의 동률 처리용 난수를 모두 고정한다.
    rng = random.Random(MAP_SEED)
    random.seed(MAP_SEED)

    agents = load_agents()
    rows = []

    for wall_prob in WALL_PROBS:
        maps, tries = sample_maps(rng, wall_prob)
        print(f"\n=== 벽 확률 p={wall_prob} (맵 생성 시도 {tries}회) ===")

        for map_class, grids in maps.items():
            if len(grids) < MAPS_PER_CLASS:
                print(f"[경고] '{CLASS_LABELS[map_class]}' 맵을 {len(grids)}개만 수집했습니다 "
                      f"(목표 {MAPS_PER_CLASS}개).")
            if not grids:
                continue

            summary_parts = []
            for version, agent in agents.items():
                results = [run_episode(agent, grid) for grid in grids]
                n = len(results)

                success_rate = sum(r["success"] for r in results) / n
                optimal_rate = sum(r["success"] and r["steps"] == OPTIMAL_STEPS
                                   for r in results) / n

                rows.append({
                    "wall_prob": wall_prob,
                    "map_class": map_class,
                    "version": version,
                    "n_maps": n,
                    "success_rate": success_rate,
                    "optimal_rate": optimal_rate,
                    "timeout_rate": sum(r["timeout"] for r in results) / n,
                    "avg_steps": sum(r["steps"] for r in results) / n,
                    "avg_collisions": sum(r["collisions"] for r in results) / n,
                })
                summary_parts.append(
                    f"{version}: 성공률={success_rate:.3f}, {OPTIMAL_STEPS}스텝 성공={optimal_rate:.3f}"
                )

            print(f"{CLASS_LABELS[map_class]} (n={len(grids)}): " + " | ".join(summary_parts))

    df = pd.DataFrame(rows)
    df.to_csv("random_map_eval.csv", index=False)
    print("\n[random_map_eval] random_map_eval.csv 저장 완료")


if __name__ == "__main__":
    main()