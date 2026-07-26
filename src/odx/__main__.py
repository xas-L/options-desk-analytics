"""Command-line interface for ODX."""

import argparse
import sys
from odx.logging import get_logger

logger = get_logger(__name__)


def run_surface(args: argparse.Namespace) -> None:
    """Run surface command."""
    logger.info("Running surface subcommand with args: %s", args)
    print("Surface subcommand executed.")


def run_cone(args: argparse.Namespace) -> None:
    """Run cone command."""
    logger.info("Running cone subcommand with args: %s", args)
    print("Cone subcommand executed.")


def run_backtest(args: argparse.Namespace) -> None:
    """Run backtest command."""
    logger.info("Running backtest subcommand with args: %s", args)
    print("Backtest subcommand executed.")


def run_walk_forward(args: argparse.Namespace) -> None:
    """Run walk-forward command."""
    logger.info("Running walk-forward subcommand with args: %s", args)
    print("Walk-forward subcommand executed.")


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="odx",
        description="Options Desk Analytics CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Surface
    parser_surface = subparsers.add_parser("surface", help="Fit or display a volatility surface")
    parser_surface.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser_surface.set_defaults(func=run_surface)

    # Cone
    parser_cone = subparsers.add_parser("cone", help="Generate a volatility cone")
    parser_cone.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser_cone.set_defaults(func=run_cone)

    # Backtest
    parser_backtest = subparsers.add_parser("backtest", help="Run a strategy backtest")
    parser_backtest.add_argument("--strategy", type=str, required=True, help="Strategy name")
    parser_backtest.set_defaults(func=run_backtest)

    # Walk-forward
    parser_wf = subparsers.add_parser("walk-forward", help="Run rolling recalibration harness")
    parser_wf.add_argument("--model", type=str, default="heston", help="Model to calibrate")
    parser_wf.set_defaults(func=run_walk_forward)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
