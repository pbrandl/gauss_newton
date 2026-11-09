"""Fit a polynomial to noisy samples of a random polynomial and plot the result.

Usage:
    python examples/polynomial_fit.py --seed 0
    python examples/polynomial_fit.py --degree 3 --n-samples 15 --output fit.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from plot_style import LINE_WIDTH, THEMES, apply_theme, style_axes

from gauss_newton import GaussNewton, Polynomial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degree", type=int, default=5, help="polynomial degree (default: 5)")
    parser.add_argument("--n-samples", type=int, default=20, help="number of samples (default: 20)")
    parser.add_argument("--noise", type=float, default=0.5, help="noise std. dev. (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light", help="color theme")
    parser.add_argument("--output", help="save the plot to this file instead of showing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theme = THEMES[args.theme]
    apply_theme(theme)
    rng = np.random.default_rng(args.seed)
    lower, upper = -1.5, 1.5

    model = Polynomial()
    true_params = rng.uniform(-2, 2, size=args.degree + 1)
    initial_params = rng.uniform(-4, 4, size=args.degree + 1)
    x = rng.uniform(lower, upper, size=args.n_samples)
    y = model(true_params, x) + rng.normal(0.0, args.noise, size=args.n_samples)

    result = GaussNewton().fit(model, x, y, initial_params)

    print(f"Observations:   {args.n_samples}")
    print(f"Converged:      {result.converged} after {result.n_iterations} iterations")
    print(f"RMSE:           {result.error:.4f}")
    print(f"Fitted model:   {model.to_string(result.params)}")
    print(f"True model:     {model.to_string(true_params)}")

    grid = np.linspace(lower, upper, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style_axes(ax, theme)
    fit_color, truth_color = theme.series
    ax.plot(grid, model(result.params, grid), color=fit_color, linewidth=LINE_WIDTH, label="Fit")
    ax.plot(grid, model(true_params, grid), color=truth_color, linewidth=LINE_WIDTH, label="Truth")
    ax.scatter(
        x,
        y,
        s=30,
        color=theme.sample,
        edgecolors=theme.surface,
        linewidths=1.0,
        label="Noisy samples",
        zorder=3,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Gauss-Newton fit of a degree-{args.degree} polynomial")
    ax.legend()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight", metadata={"Date": None})
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
