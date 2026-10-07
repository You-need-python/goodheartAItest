"""
표 기반(tabular) Q-learning 에이전트.

상태 표현 (경로 A 반영):
    state = (wall_up, wall_down, wall_left, wall_right)

grid_env.GridWorldEnv 가 반환하는 이 6-튜플을 그대로 Q-table의 인덱스로
사용한다. Q-table의 형태는 다음과 같다.

    (2, 2, 2, 2, num_actions)

--앞의 두 축(n_rows, n_cols)은 절대 좌표(삭제함)--
가운데 네 축은 각각 0/1 값을
가지는 상/하/좌/우 벽 감지 비트, 마지막 축은 행동이다. 행동은
config.ACTIONS 에 정의된 문자열이며, 이 리스트의 인덱스 순서를 Q-table의
마지막 축과 그대로 대응시킨다.
"""

import random

import numpy as np

from config import ACTIONS, ALPHA, GAMMA, NUM_ACTIONS


class QLearningAgent:
    def __init__(self, grid_shape, alpha=ALPHA, gamma=GAMMA):
        """
        Args:
            grid_shape: (n_rows, n_cols) - 환경의 격자 크기
            alpha: 학습률
            gamma: 할인율
        """
        n_rows, n_cols = grid_shape
        # 앞의 두 축(n_rows, n_cols) 뒤에 이어지는 (2,2,2,2)는 상/하/좌/우 벽 감지 비트.
        self.q_table = np.zeros((2, 2, 2, 2, NUM_ACTIONS), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state, epsilon):
        """
        ε-greedy 정책에 따라 행동을 선택한다.

        Args:
            state: (wall_up, wall_down, wall_left, wall_right)
            epsilon: 무작위 행동을 선택할 확률. 0이면 완전 탐욕적(평가 시 사용).

        Returns:
            action: config.ACTIONS 중 하나의 문자열
        """
        if random.random() < epsilon:
            return random.choice(ACTIONS)

        # state가 4-튜플이므로 numpy 다차원 인덱싱에 그대로 사용하면
        # 마지막 축(행동)에 대한 값 배열이 반환된다.
        action_values = self.q_table[state]
        best_indices = np.flatnonzero(action_values == action_values.max())
        # 최댓값이 동률인 행동이 여러 개면 무작위로 하나를 골라 편향을 방지한다.
        best_index = random.choice(list(best_indices))
        return ACTIONS[best_index]

    def update(self, state, action, reward, next_state, done):
        """
        표준 Q-learning 갱신 규칙을 적용한다.

            Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        done이 True이면(에피소드 종료) 다음 상태의 가치를 0으로 취급한다.

        Args:
            state, next_state: (wall_up, wall_down, wall_left, wall_right)
            action: config.ACTIONS 중 하나의 문자열
        """
        a_idx = ACTIONS.index(action)
        idx = state + (a_idx,)  # 4-튜플 + 행동 인덱스 = 5-튜플 (Q-table의 완전한 인덱스)

        current_q = self.q_table[idx]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state].max()

        self.q_table[idx] = current_q + self.alpha * (target - current_q)

    def save(self, path):
        """Q-table을 path에 저장한다 (numpy .npy 형식)."""
        np.save(path, self.q_table)

    def load(self, path):
        """path에 저장된 Q-table을 불러와 현재 에이전트에 덮어쓴다."""
        self.q_table = np.load(path)