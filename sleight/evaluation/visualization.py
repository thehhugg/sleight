"""Visualization utilities for robustness evaluation results."""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_accuracy_vs_epsilon(
    results: list[dict[str, float]],
    title: str = "Accuracy vs. Perturbation Budget",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot clean and adversarial accuracy as a function of epsilon.

    Args:
        results: Output of ``epsilon_sweep`` — a list of dicts, each
            containing ``epsilon``, ``clean_accuracy``, and
            ``adversarial_accuracy``.
        title: Plot title.
        save_path: If provided, save the figure to this file path.
        show: If ``True``, call ``plt.show()``.

    Returns:
        The matplotlib ``Figure`` object.
    """
    epsilons = [r["epsilon"] for r in results]
    clean_acc = [r["clean_accuracy"] for r in results]
    adv_acc = [r["adversarial_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, clean_acc, "o-", label="Clean Accuracy", linewidth=2)
    ax.plot(epsilons, adv_acc, "s-", label="Adversarial Accuracy", linewidth=2)
    ax.set_xlabel("Epsilon (perturbation budget)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_attack_comparison(
    results_dict: dict[str, list[dict[str, float]]],
    metric: str = "adversarial_accuracy",
    title: str = "Attack Comparison",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Compare multiple attacks on the same plot.

    Args:
        results_dict: A dict mapping attack names to their ``epsilon_sweep``
            results.
        metric: Which metric to plot on the y-axis. Defaults to
            ``adversarial_accuracy``.
        title: Plot title.
        save_path: If provided, save the figure to this file path.
        show: If ``True``, call ``plt.show()``.

    Returns:
        The matplotlib ``Figure`` object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, results in results_dict.items():
        epsilons = [r["epsilon"] for r in results]
        values = [r[metric] for r in results]
        ax.plot(epsilons, values, "o-", label=name, linewidth=2)

    ax.set_xlabel("Epsilon (perturbation budget)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
