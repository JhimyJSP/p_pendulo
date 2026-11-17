"""Interfaz de linea de comandos: generate, train, evaluate, all, sweep."""

import argparse
from pathlib import Path

import pandas as pd

from .config import Config, ROOT
from .data import generate_data
from .experiment import MODELS, run_evaluation, run_training


def parser_factory() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pendulo", description="Comparacion reproducible de modelos para el pendulo ideal")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "train", "evaluate", "all", "sweep"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--data-dir", type=Path, default=ROOT / "data")
        cmd.add_argument("--output-dir", type=Path, default=ROOT / ("data" if command == "generate" else "results"))
        if command == "generate":
            cmd.add_argument("--overwrite", action="store_true")
            continue
        cmd.add_argument("--epochs", type=int, default=1200)
        cmd.add_argument("--patience", type=int, default=120)
        cmd.add_argument("--batch-size", type=int, default=2048)
        cmd.add_argument("--hidden-dim", type=int, default=64)
        cmd.add_argument("--learning-rate", type=float, default=1e-3)
        cmd.add_argument("--val-fraction", type=float, default=0.10)
        cmd.add_argument("--seed", type=int, default=2026)
        cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        cmd.add_argument("--print-every", type=int, default=50)
        if command in ("train", "all", "sweep"):
            cmd.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                         help="Modelos a entrenar (por defecto, ambos)")
        if command == "sweep":
            cmd.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200],
                             choices=[100, 150, 200])
        else:
            cmd.add_argument("--n-train", type=int, choices=[100, 150, 200], default=200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parser_factory().parse_args(argv)
    if args.command == "generate":
        generate_data(args.data_dir, args.output_dir, Config(), overwrite=args.overwrite)
        return
    sizes = args.sizes if args.command == "sweep" else [args.n_train]
    summaries = []
    for n in sizes:
        config = Config(
            n_train=n, max_epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate, val_fraction=args.val_fraction,
            seed=args.seed, device=args.device, print_every=args.print_every,
        )
        if args.command in ("train", "all", "sweep"):
            run_training(args.data_dir, args.output_dir, config, tuple(dict.fromkeys(args.models)))
        if args.command in ("evaluate", "all", "sweep"):
            metrics = run_evaluation(args.data_dir, args.output_dir, config)
            for kind, values in metrics.items():
                summaries.append({"n_train": n, "model": kind, **values})
    if args.command == "sweep":
        dest = args.output_dir / f"comparison_{'_'.join(f'N{n}' for n in sizes)}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(dest, index=False)
        print(f"Comparacion entre tamanos guardada en: {dest}")
        domain_tables = [pd.read_csv(args.output_dir / f"N{n}" / "metrics_test_domains.csv")
                         .assign(n_train=n) for n in sizes]
        pd.concat(domain_tables, ignore_index=True).to_csv(
            dest.with_stem(f"{dest.stem}_domains"), index=False)


if __name__ == "__main__":
    main()
