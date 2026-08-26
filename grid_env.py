"""
그리드월드 환경.

핵심 설계 원칙:
    이 환경은 보상을 전혀 계산하지 않는다. step()은 오직 다음 상태와
    info(딕셔너리)만을 반환하며, 보상은 reward_configs.py 의 함수가
    info 를 입력받아 환경 바깥에서 별도로 계산한다.

    이렇게 분리해야 "에이전트가 학습 중 최적화하는 대리 보상(proxy reward)"
    과 "실제 목표를 나타내는 원시 관측값(info)"이 코드 구조상으로도
    명확히 분리되고, evaluate.py 는 이 info 만으로 실제 목표 달성도를
    측정할 수 있다.
"""

from config import ACTION_DELTAS, MAX_STEPS_PER_EPISODE


class GridWorldEnv:
    def __init__(self, grid, start, goal, max_steps=MAX_STEPS_PER_EPISODE):
        """
        Args:
            grid: np.ndarray[bool], shape=(n_rows, n_cols). True인 칸은 벽.
            start: (row, col)
            goal: (row, col)
            max_steps: 한 에피소드의 최대 스텝 수 (초과 시 timeout으로 종료)
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.n_rows, self.n_cols = grid.shape
        self.max_steps = max_steps

        self.position = None
        self.visited = None
        self.steps_taken = 0

    def reset(self):
        """에피소드를 시작 지점에서 새로 시작한다."""
        self.position = self.start
        self.visited = {self.start}
        self.steps_taken = 0
        return self.position

    def _manhattan(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, action):
        """
        Args:
            action: config.ACTIONS 중 하나의 문자열 ("UP"/"DOWN"/"LEFT"/"RIGHT")

        Returns:
            next_state: (row, col)
            info: dict — 아래 키를 포함하며, 보상 계산과 평가에 필요한
                모든 원시 데이터를 담는다.
                    position       : 이동 후 위치
                    prev_distance  : 이동 전 목적지까지 맨해튼 거리
                    curr_distance  : 이동 후 목적지까지 맨해튼 거리
                    collided       : 벽/경계에 부딪혀 제자리에 머물렀는지 여부
                    revisited      : 이번에 도달한 칸을 이전에 방문한 적이 있는지 여부
                    steps_taken    : 이번 에피소드에서 지금까지 이동한 스텝 수
                    success        : 목적지에 도달했는지 여부
                    timeout        : 최대 스텝 수를 초과해 강제 종료되었는지 여부
            done: bool — success 또는 timeout 이면 True
        """
        if self.position is None:
            raise RuntimeError("step()을 호출하기 전에 reset()을 먼저 호출해야 합니다.")

        prev_distance = self._manhattan(self.position)

        dr, dc = ACTION_DELTAS[action]
        candidate = (self.position[0] + dr, self.position[1] + dc)

        collided = False
        if not self._in_bounds(candidate) or self.grid[candidate]:
            # 격자 밖이거나 벽인 칸으로 이동을 시도한 경우: 위치는 그대로 유지
            collided = True
            candidate = self.position

        revisited = candidate in self.visited

        self.position = candidate
        self.visited.add(candidate)
        self.steps_taken += 1

        curr_distance = self._manhattan(self.position)
        success = self.position == self.goal
        timeout = (not success) and (self.steps_taken >= self.max_steps)
        done = success or timeout

        info = {
            "position": self.position,
            "prev_distance": prev_distance,
            "curr_distance": curr_distance,
            "collided": collided,
            "revisited": revisited,
            "steps_taken": self.steps_taken,
            "success": success,
            "timeout": timeout,
        }

        return self.position, info, done