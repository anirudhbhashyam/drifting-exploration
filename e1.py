# /// script
# requires-python  = ">=3.14"
# dependencies = [
#     "matplotlib",
# ]
# ///

import argparse
import itertools
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt

type DriftingField = Callable[[Point, Point], Point]


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    @classmethod
    def zero(cls) -> Self:
        return cls(0, 0)

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def dist(self, other: Self) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def is_zero(self) -> bool:
        return self == type(self)(0.0, 0.0)

    def __add__(self, other: Self) -> Self:
        return type(self)(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return type(self)(self.x - other.x, self.y - other.y)

    def __eq__(self, other: Self) -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


def drifting_field(p1: Point, p2: Point) -> Point:
    if p1 == p2:
        return p1
    scale = 1 / p1.dist(p2)
    return Point(scale * (p1.x - p2.x), scale * (p1.y - p2.y))


def visualize_points(points: list[Point]) -> plt.Figure:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        [p.x for p in points],
        [p.y for p in points],
        c="#28a0fc",
        edgecolor="none",
        alpha=0.8,
        s=20,
    )
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def update_step(points: list[Point], field: DriftingField) -> list[Point]:
    return [point - field(point, Point.zero()) for point in points]


def process_args() -> argparse.Namespace:
    processor = argparse.ArgumentParser()
    _ = processor.add_argument(
        "--n-steps",
        "-n",
        type=int,
        default=200,
        help="The number of simulation steps.",
    )
    _ = processor.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed value to use.",
    )
    return processor.parse_args()


def main(args: argparse.Namespace) -> int:
    # Points collapse to the origin
    n_steps = args.n_steps
    random.seed(args.seed)
    points = [
        Point(x, y) for x, y in itertools.product(range(-100, 100), range(-100, 100))
    ]
    selected_points = random.choices(points, k=100)
    out_path = Path(__file__).parent.joinpath("e1_data")
    if not out_path.exists():
        out_path.mkdir(parents=True)
    for step in range(n_steps):
        selected_points = update_step(selected_points, drifting_field)
        fig = visualize_points(selected_points)
        fig.savefig(out_path.joinpath(f"step_{step}.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    args = process_args()
    raise SystemExit(main(args))
