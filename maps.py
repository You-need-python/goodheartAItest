"""
격자 맵 정의 및 파싱 유틸리티.

맵 표기 규칙:
    '.' : 빈 칸
    '#' : 벽(장애물)
    'S' : 시작 지점
    'G' : 목적지

TRAIN_MAP 은 학습(train.py)에 사용하고,
HOLDOUT_MAP 은 학습에는 절대 사용하지 않으며 evaluate.py 의
일반화(generalization) 평가 전용으로만 사용한다.
두 맵은 같은 크기를 갖되 벽의 배치가 서로 다르다.
"""

from collections import deque

import numpy as np

TRAIN_MAP = [
    "S..#....",
    "..#.....",
    "..#.##..",
    "....#...",
    ".##.#.#.",
    "......#.",
    ".####...",
    "......#G",
]

HOLDOUT_MAP = [
    "S.#.....",
    ".##.#...",
    "......#.",
    ".#.###..",
    ".#......",
    "..#.##.#",
    "....#...",
    "..#....G",
]


def parse_map(map_rows):
    """
    문자열 리스트 형태의 맵을 numpy 배열과 시작/목적지 좌표로 변환한다.

    Args:
        map_rows: 문자열의 리스트 (각 문자열은 한 행)

    Returns:
        grid: np.ndarray[bool], shape=(n_rows, n_cols). True인 칸은 벽.
        start: (row, col)
        goal: (row, col)
        shape: (n_rows, n_cols)
    """
    n_rows = len(map_rows)
    n_cols = len(map_rows[0])

    for row in map_rows:
        if len(row) != n_cols:
            raise ValueError("맵의 모든 행은 같은 길이를 가져야 합니다.")

    grid = np.zeros((n_rows, n_cols), dtype=bool)
    start = None
    goal = None

    for r, row in enumerate(map_rows):
        for c, ch in enumerate(row):
            if ch == "#":
                grid[r, c] = True
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch == ".":
                pass
            else:
                raise ValueError(f"알 수 없는 맵 문자: {ch!r} (위치: {r}, {c})")

    if start is None:
        raise ValueError("맵에 시작 지점(S)이 없습니다.")
    if goal is None:
        raise ValueError("맵에 목적지(G)가 없습니다.")

    return grid, start, goal, (n_rows, n_cols)


def is_reachable(grid, start, goal):
    """
    BFS로 start에서 goal까지 4방향 이동만으로 도달 가능한지 확인한다.
    맵 설계 단계에서 목적지가 벽에 막혀 도달 불가능한 실수를 막기 위한
    검증용 함수이며, 학습/평가 로직에서는 사용하지 않는다.
    """
    n_rows, n_cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])
    visited[start] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                if not grid[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return False


if __name__ == "__main__":
    # 모듈을 직접 실행하면 두 맵 모두에 대해 시작->목적지 도달 가능성을 검증한다.
    for name, map_data in [("TRAIN_MAP", TRAIN_MAP), ("HOLDOUT_MAP", HOLDOUT_MAP)]:
        grid, start, goal, shape = parse_map(map_data)
        reachable = is_reachable(grid, start, goal)
        print(f"{name}: shape={shape}, start={start}, goal={goal}, reachable={reachable}")
        assert reachable, f"{name}에서 목적지에 도달할 수 없습니다. 맵을 수정하세요."