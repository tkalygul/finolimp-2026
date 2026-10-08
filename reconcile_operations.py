"""Command-line entry point for Step 2 (balance bridge is Step 3)."""
import argparse
from src.matching import run_matching


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сопоставление 1С, ETM и реестра")
    parser.add_argument("--clean", default="interim/clean")
    parser.add_argument("--registry", default="interim/registry.parquet")
    parser.add_argument("--out", default="interim/reconciliation")
    args = parser.parse_args()
    run_matching(args.clean, args.registry, args.out)
