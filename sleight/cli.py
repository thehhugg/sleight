"""Command-line interface for sleight."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv: list[str] | None = None) -> None:
    """Entry point for the sleight CLI."""
    parser = argparse.ArgumentParser(
        prog="sleight",
        description="Adversarial ML toolkit — attack, defend, and evaluate neural networks.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- attack ---
    attack_parser = subparsers.add_parser("attack", help="Run an adversarial attack")
    attack_parser.add_argument(
        "method",
        choices=["fgsm", "pgd", "cw", "deepfool"],
        help="Attack method to use",
    )
    attack_parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to use (default: mnist)",
    )
    attack_parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Perturbation budget (default: 0.1)"
    )
    attack_parser.add_argument(
        "--samples", type=int, default=100, help="Number of test samples (default: 100)"
    )
    attack_parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs for the model (default: 3)"
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model robustness across epsilon values"
    )
    eval_parser.add_argument(
        "method",
        choices=["fgsm", "pgd", "cw", "deepfool"],
        help="Attack method to evaluate against",
    )
    eval_parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to use (default: mnist)",
    )
    eval_parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        help="Epsilon values to sweep (default: 0.0 0.05 0.1 0.15 0.2 0.25 0.3)",
    )
    eval_parser.add_argument(
        "--samples", type=int, default=200, help="Number of test samples (default: 200)"
    )
    eval_parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs for the model (default: 3)"
    )
    eval_parser.add_argument(
        "--save-plot", type=str, default=None, help="Save accuracy-vs-epsilon plot to file"
    )

    # --- list ---
    subparsers.add_parser("list", help="List available attacks and defenses")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        _cmd_list()
    elif args.command == "attack":
        _cmd_attack(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)


def _cmd_list() -> None:
    """List available attacks and defenses."""
    print("Attacks:")
    print("  fgsm       Fast Gradient Sign Method (Goodfellow et al., 2015)")
    print("  pgd        Projected Gradient Descent (Madry et al., 2018)")
    print("  cw         Carlini & Wagner L2 (Carlini & Wagner, 2017)")
    print("  deepfool   DeepFool (Moosavi-Dezfooli et al., 2016)")
    print()
    print("Defenses:")
    print("  adversarial_training    Train on clean + adversarial examples")
    print("  jpeg_compression        JPEG lossy compression")
    print("  spatial_smoothing       Mean filter smoothing")
    print("  bit_depth_reduction     Pixel quantization")
    print("  defensive_distillation  Teacher-student distillation")
    print()
    print("Datasets:")
    print("  mnist           MNIST handwritten digits (28x28x1)")
    print("  fashion_mnist   Fashion-MNIST (28x28x1)")
    print("  cifar10         CIFAR-10 (32x32x3)")


def _get_attack_fn(method: str):
    """Return the attack function for the given method name."""
    from sleight.attacks import fgsm_attack, pgd_attack, cw_attack, deepfool_attack

    attacks = {
        "fgsm": fgsm_attack,
        "pgd": pgd_attack,
        "cw": cw_attack,
        "deepfool": deepfool_attack,
    }
    return attacks[method]


def _get_model_and_data(dataset: str, samples: int, epochs: int):
    """Load data, build model, train briefly, and return model + test subset."""
    import tensorflow as tf
    from sleight.data import get_dataset

    (x_train, y_train), (x_test, y_test) = get_dataset(dataset, one_hot=True)

    if dataset == "cifar10":
        from sleight.models import get_cifar10_cnn_model

        model = get_cifar10_cnn_model()
    else:
        from sleight.models import get_mnist_cnn_model

        model = get_mnist_cnn_model()

    print(f"Training model on {dataset} for {epochs} epoch(s)...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1)

    x_sub = x_test[:samples]
    y_sub = y_test[:samples]
    # Integer labels for evaluation
    y_sub_int = np.argmax(y_sub, axis=1)

    return model, x_sub, y_sub, y_sub_int


def _cmd_attack(args) -> None:
    """Run an adversarial attack and report results."""
    model, x_sub, y_sub, y_sub_int = _get_model_and_data(
        args.dataset, args.samples, args.epochs
    )
    attack_fn = _get_attack_fn(args.method)

    print(f"\nRunning {args.method.upper()} attack (epsilon={args.epsilon})...")
    adv_images = np.array(attack_fn(model, x_sub, y_sub, args.epsilon))

    clean_preds = np.argmax(model.predict(x_sub, verbose=0), axis=1)
    adv_preds = np.argmax(model.predict(adv_images, verbose=0), axis=1)

    clean_acc = np.mean(clean_preds == y_sub_int)
    adv_acc = np.mean(adv_preds == y_sub_int)
    clean_correct = clean_preds == y_sub_int
    adv_correct = adv_preds == y_sub_int
    success_rate = (
        np.sum(clean_correct & ~adv_correct) / max(np.sum(clean_correct), 1)
    )

    print(f"\nResults ({args.method.upper()}, epsilon={args.epsilon}):")
    print(f"  Clean accuracy:       {clean_acc:.4f}")
    print(f"  Adversarial accuracy: {adv_acc:.4f}")
    print(f"  Attack success rate:  {success_rate:.4f}")


def _cmd_evaluate(args) -> None:
    """Run epsilon sweep and optionally save plot."""
    model, x_sub, y_sub, y_sub_int = _get_model_and_data(
        args.dataset, args.samples, args.epochs
    )
    attack_fn = _get_attack_fn(args.method)

    from sleight.evaluation import epsilon_sweep, plot_accuracy_vs_epsilon

    print(f"\nRunning epsilon sweep for {args.method.upper()}...")
    results = epsilon_sweep(model, x_sub, y_sub_int, attack_fn, args.epsilons)

    print(f"\n{'Epsilon':>10} {'Clean Acc':>12} {'Adv Acc':>12} {'Success Rate':>14}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['epsilon']:>10.3f} {r['clean_accuracy']:>12.4f} "
            f"{r['adversarial_accuracy']:>12.4f} {r['attack_success_rate']:>14.4f}"
        )

    if args.save_plot:
        fig = plot_accuracy_vs_epsilon(
            results,
            title=f"{args.method.upper()} on {args.dataset}",
            save_path=args.save_plot,
        )
        print(f"\nPlot saved to {args.save_plot}")


if __name__ == "__main__":
    main()
