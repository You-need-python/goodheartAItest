"""
그리드월드 환경.

핵심 설계 원칙 (보상-환경 분리):
    이 환경은 보상을 전혀 계산하지 않는다. step()은 오직 다음 상태와
    info(딕셔너리)만을 반환하며, 보상은 reward_configs.py 의 함수가
    info 를 입력받아 환경 바깥에서 별도로 계산한다.

상태 표현 (경로 A 반영):
    state = (row, col, wall_up, wall_down, wall_left, wall_right)

    (row, col)는 절대 좌표이고, wall_up/down/left/right는 현재 위치를
    기준으로 상/하/좌/우 방향에 벽(또는 격자 경계)이 있는지를 나타내는
    0/1 값이다.

    좌표를 여전히 포함하는 이유: 이 프로젝트의 모든 맵에서 목적지는 항상
    같은 좌표(우하단 모서리)에 고정되어 있으므로, "목적지까지 얼마나
    남았는가"라는 방향 정보는 사실상 좌표 자체에 담겨 있다. 좌표를 완전히
    제거하면 에이전트는 목적지 방향을 전혀 알 수 없어, 벽을 피하는 것
    이상의 목표 지향적 행동을 학습할 수 없다.

    벽 감지 비트를 추가하는 이유: 여러 맵을 하나의 Q-table로 함께
    학습시킬 때, 같은 좌표라도 맵마다 벽 배치가 다르면 서로 다른 학습
    신호가 같은 테이블 칸에 충돌해서 누적되는 문제가 있었다(다중 맵
    학습 실험에서 실제로 확인됨). 벽 감지 비트를 상태에 포함시키면,
    같은 좌표라도 국소 벽 구조가 다르면 별도의 상태로 구분되므로 이
    충돌이 해소된다.

    주의(이 설계가 보장하지 않는 것): 이 변경은 "여러 맵을 동시에
    학습해도 서로의 학습 신호를 훼손하지 않도록" 만드는 것이며, 좌표가
    상태에 그대로 남아있는 이상 "완전히 새로운 맵의 낯선 좌표-벽 조합"에
    대해 자동으로 일반화를 보장하지는 않는다. 이는 재학습 후 실제
    홀드아웃 평가로 별도 확인이 필요하다.
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
        return self._make_state(self.position)

    def _manhattan(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_wall(self, pos):
        """격자 밖이거나 벽이면 True. 경계도 '벽'으로 취급해 4방향 감지가 항상 정의되도록 한다."""
        if not self._in_bounds(pos):
            return True
        return bool(self.grid[pos])

    def _local_walls(self, pos):
        """pos를 기준으로 상/하/좌/우에 벽(또는 경계)이 있는지를 0/1로 반환한다."""
        r, c = pos
        wall_up = self._is_wall((r - 1, c))
        wall_down = self._is_wall((r + 1, c))
        wall_left = self._is_wall((r, c - 1))
        wall_right = self._is_wall((r, c + 1))
        return int(wall_up), int(wall_down), int(wall_left), int(wall_right)

    def _make_state(self, pos):
        """(row, col, wall_up, wall_down, wall_left, wall_right) 튜플을 만든다."""
        wu, wd, wl, wr = self._local_walls(pos)
        return (pos[0], pos[1], wu, wd, wl, wr)

    def step(self, action):
        """
        Args:
            action: config.ACTIONS 중 하나의 문자열 ("UP"/"DOWN"/"LEFT"/"RIGHT")

        Returns:
            next_state: (row, col, wall_up, wall_down, wall_left, wall_right)
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

        next_state = self._make_state(self.position)
        return next_state, info, done