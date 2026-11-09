"""Fit a damped oscillation with Gauss-Newton and plot how the fit evolves per iteration.

The figures in the README were generated with:
    python examples/damped_oscillation.py --theme light --output docs/convergence-light.svg
    python examples/damped_oscillation.py --theme dark --output docs/convergence-dark.svg
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from plot_style import LINE_WIDTH, MARKER_SIZE, RING_WIDTH, THEMES, apply_theme, style_axes

from gauss_newton import GaussNewton

TRUE_PARAMS = (2.0, 0.4, 2.2)
INITIAL_PARAMS = (1.0, 0.2, 1.8)


def damped_oscillation(params: ArrayLike, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``a * exp(-b * x) * cos(omega * x)`` with ``params = (a, b, omega)``."""
    amplitude, decay, frequency = np.asarray(params, dtype=float)
    return amplitude * np.exp(-decay * x) * np.cos(frequency * x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=60, help="number of samples (default: 60)")
    parser.add_argument("--noise", type=float, default=0.12, help="noise std. dev. (default: 0.12)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light", help="color theme")
    parser.add_argument("--output", help="save the plot to this file instead of showing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theme = THEMES[args.theme]
    apply_theme(theme)

    rng = np.random.default_rng(args.seed)
    x = np.linspace(0.0, 8.0, args.n_samples)
    y = damped_oscillation(TRUE_PARAMS, x) + rng.normal(0.0, args.noise, size=x.size)

    result = GaussNewton().fit(damped_oscillation, x, y, INITIAL_PARAMS)
    history = np.asarray(result.history)

    print(f"Converged:  {result.converged} after {result.n_iterations} iterations")
    print(f"RMSE:       {result.error:.4f}")
    print(
        f"Parameters: a={result.params[0]:.3f}, b={result.params[1]:.3f}, "
        f"omega={result.params[2]:.3f}  (true: {TRUE_PARAMS})"
    )

    # Draw the iterations up to the point where the fit stops visibly changing, plus the
    # final fit. Each gets the next step of the blue ramp; later iterations look the same
    # as the final fit and share its color.
    last = result.n_iterations
    settled = int(np.argmax(history <= result.error * 1.01))
    n_steps = min(settled, len(theme.ramp) - 1)
    shown = [*np.linspace(0, settled - 1, n_steps).round().astype(int), last] if n_steps else [last]
    colors = dict(zip(shown, theme.ramp[-len(shown) :], strict=True))

    def color_of(iteration: int) -> str:
        if iteration >= settled:
            return colors[last]
        return colors[max(i for i in shown if i <= iteration)]

    fig, (ax_fit, ax_err) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.85, 1], "wspace": 0.18}
    )

    # Left: samples and the model after selected iterations.
    style_axes(ax_fit, theme)
    grid = np.linspace(x.min(), x.max(), 500)
    ax_fit.scatter(
        x,
        y,
        s=30,
        color=theme.sample,
        edgecolors=theme.surface,
        linewidths=1.0,
        label="Samples",
        zorder=2,
    )
    for i in shown:
        if i == last:
            label = f"Final fit (iteration {i})"
        else:
            label = "Initial guess" if i == 0 else f"Iteration {i}"
        ax_fit.plot(
            grid,
            damped_oscillation(result.params_history[i], grid),
            color=colors[i],
            linewidth=LINE_WIDTH,
            label=label,
            zorder=3 + i,
        )
    ax_fit.set_title("Model after each Gauss-Newton step")
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.set_xlim(x.min() - 0.1, x.max() + 0.1)
    low, high = ax_fit.get_ylim()
    ax_fit.set_ylim(low, high + 0.15 * (high - low))  # headroom for the legend
    ax_fit.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)

    # Right: error per iteration, markers colored like the curves on the left.
    style_axes(ax_err, theme)
    iterations = np.arange(len(history))
    ax_err.plot(iterations, history, color=theme.axis, linewidth=LINE_WIDTH, zorder=2)
    ax_err.scatter(
        iterations,
        history,
        s=MARKER_SIZE**2,
        zorder=3,
        color=[color_of(i) for i in iterations],
        edgecolors=theme.surface,
        linewidths=RING_WIDTH,
    )
    for i in (0, len(history) - 1):
        ax_err.annotate(
            f"{history[i]:.3f}",
            (i, history[i]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=theme.text_secondary,
            fontsize=9,
        )
    ax_err.set_title("RMSE per iteration")
    ax_err.set_xlabel("Iteration")
    ax_err.set_xticks(iterations)
    ax_err.set_ylim(0, history.max() * 1.18)
    ax_err.set_xlim(-0.5, len(history) - 0.5)

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
