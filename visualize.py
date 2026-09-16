"""
train_log.csv, eval_log.csv 를 시각화한다.

- plot_learning_curves() : 버전별 "에피소드당 누적 대리 보상"의 이동평균 곡선
- plot_success_curves()  : 버전별 "에피소드당 실제 성공 여부"의 이동평균(=구간 성공률) 곡선
    두 곡선을 나란히 놓고 비교하면, 대리 보상은 상승하는데 실제 성공률은
    그만큼 오르지 않는 구간이 있는지를 시각적으로 확인할 수 있다.
- plot_eval_comparison() : eval_log.csv 기반, 버전 x 맵(train/holdout)별
    성공률·충돌 횟수·스텝 수 막대그래프
- plot_trajectory()      : 저장된 Q-table로 완전 탐욕적 정책을 한 번 굴려,
    실제 이동 경로를 격자 위에 그린다.
    Pygame 대신 matplotlib으로 구현했다. 디스플레이가 없는 환경(서버, 코드
    실행 환경 등)에서도 동일하게 PNG로 저장되어 동작한다. 실시간 애니메이션이
    필요하면 이 함수의 좌표 기록(trace) 로직을 그대로 pygame 렌더링 루프에
    옮겨 재사용할 수 있다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 그래프의 한글 레이블(축 이름, 제목, 범례 등)이 깨지지 않도록
# 시스템에 설치된 한글 지원 폰트를 matplotlib 기본 폰트로 지정한다.
# 참고: matplotlib의 폰트 매니저는 .ttc(트루타입 컬렉션) 파일 하나당
# 첫 번째 서브폰트 이름만 등록하는 경우가 많아, "Noto Sans CJK KR"이
# 목록에 없더라도 같은 파일에 포함된 "Noto Sans CJK JP" 등의 이름으로
# 등록되어 있을 수 있다. 이 경우에도 한글(한글 자모)은 정상적으로 포함되어
# 있으므로, 이름에 "CJK"가 들어간 폰트를 폭넓게 후보로 탐색한다.
_KOREAN_FONT_CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}

_chosen_font = next((name for name in _KOREAN_FONT_CANDIDATES if name in _available_fonts), None)
if _chosen_font is None:
    _chosen_font = next((name for name in sorted(_available_fonts) if "CJK" in name), None)

if _chosen_font is not None:
    plt.rcParams["font.family"] = _chosen_font
else:
    print("[visualize] 경고: 한글 지원 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")

plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 마이너스 기호 깨짐 방지

from grid_env import GridWorldEnv
from maps import TRAIN_MAP, HOLDOUT_MAP, parse_map
from q_learning_agent import QLearningAgent

VERSIONS = ["v1", "v2", "v3"]
VERSION_COLORS = {"v1": "tab:red", "v2": "tab:blue", "v3": "tab:green"}


def _moving_average(values, window=20):
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_learning_curves(log_path="train_log.csv", save_path="learning_curves.png", window=20):
    """버전별 에피소드당 누적 대리 보상의 이동평균 곡선을 그린다."""
    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for version in VERSIONS:
        sub = df[df["version"] == version].sort_values("episode")
        smoothed = _moving_average(sub["cumulative_proxy_reward"].to_numpy(), window)
        ax.plot(sub["episode"], smoothed, label=version, color=VERSION_COLORS[version])

    ax.set_xlabel("에피소드")
    ax.set_ylabel(f"누적 대리 보상 (이동평균, window={window})")
    ax.set_title("버전별 학습 곡선 - 대리 보상(proxy reward)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] {save_path} 저장 완료")


def plot_success_curves(log_path="train_log.csv", save_path="success_curves.png", window=20):
    """
    버전별 에피소드당 실제 성공 여부의 이동평균(=구간 성공률) 곡선을 그린다.
    plot_learning_curves() 의 결과와 나란히 놓고 비교하기 위한 짝 함수이다.
    """
    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for version in VERSIONS:
        sub = df[df["version"] == version].sort_values("episode")
        success_numeric = sub["success"].astype(int).to_numpy()
        smoothed = _moving_average(success_numeric, window)
        ax.plot(sub["episode"], smoothed, label=version, color=VERSION_COLORS[version])

    ax.set_xlabel("에피소드")
    ax.set_ylabel(f"실제 성공률 (이동평균, window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("버전별 학습 곡선 - 실제 목표 달성도(성공률)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] {save_path} 저장 완료")


def plot_eval_comparison(log_path="eval_log.csv", save_path="eval_comparison.png"):
    """
    eval_log.csv 기반으로 버전 x 맵(train/holdout)별
    성공률/평균 충돌 횟수/평균 스텝 수를 막대그래프로 비교한다.
    """
    df = pd.read_csv(log_path)

    metrics = ["success_rate", "avg_collisions", "avg_steps"]
    titles = ["성공률", "평균 충돌 횟수", "평균 스텝 수"]
    map_types = list(df["map_type"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(VERSIONS))
    bar_width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        for i, map_type in enumerate(map_types):
            sub = df[df["map_type"] == map_type].set_index("version").reindex(VERSIONS)
            offset = (i - (len(map_types) - 1) / 2) * bar_width
            ax.bar(x + offset, sub[metric], width=bar_width, label=map_type)

        ax.set_xticks(x)
        ax.set_xticklabels(VERSIONS)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

    axes[0].legend(title="맵 종류")
    fig.suptitle("버전별 · 맵별 실제 목표 지표 비교 (evaluate.py 결과)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] {save_path} 저장 완료")


def plot_trajectory(version, map_name="train", save_path=None):
    """
    저장된 Q-table로 완전 탐욕적 정책(ε=0)을 한 번 굴려, 실제 이동 경로를
    격자 위에 그린다.

    Args:
        version: "v1" / "v2" / "v3"
        map_name: "train" 또는 "holdout"
        save_path: 저장 경로 (기본값: trajectory_{version}_{map_name}.png)
    """
    map_data = TRAIN_MAP if map_name == "train" else HOLDOUT_MAP
    grid, start, goal, shape = parse_map(map_data)

    agent = QLearningAgent(grid_shape=shape)
    agent.load(f"q_table_{version}.npy")

    env = GridWorldEnv(grid, start, goal)
    state = env.reset()
    trace = [state]

    done = False
    info = None
    while not done:
        action = agent.choose_action(state, epsilon=0.0)
        state, info, done = env.step(action)
        trace.append(state)

    if save_path is None:
        save_path = f"trajectory_{version}_{map_name}.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1)

    rows = [p[0] for p in trace]
    cols = [p[1] for p in trace]
    ax.plot(cols, rows, marker="o", markersize=4, linewidth=1.5, color="tab:orange")

    ax.scatter([start[1]], [start[0]], marker="s", s=120, color="tab:blue", label="시작(S)")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=200, color="tab:green", label="목적지(G)")

    ax.set_title(
        f"{version} 정책의 이동 경로 ({map_name} 맵)\n"
        f"성공={info['success']}, 스텝={info['steps_taken']}"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] {save_path} 저장 완료")


def main():
    plot_learning_curves()
    plot_success_curves()
    plot_eval_comparison()

    for version in VERSIONS:
        plot_trajectory(version, map_name="train")
        plot_trajectory(version, map_name="holdout")


if __name__ == "__main__":
    main()