#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(AA_ALPHABET, k=length))


def yaml_msa_path(path: Path) -> str:
    path_text = path.as_posix()
    return path_text if path.is_absolute() else f"./{path_text}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a random single-chain tt-boltz YAML input and matching random A3M."
    )
    parser.add_argument("length", type=int, help="Protein sequence length.")
    parser.add_argument(
        "--msa-seqs",
        type=int,
        default=1024,
        help="Total number of MSA sequences to write, including the query.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename and target name. Defaults to the sequence length.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples"),
        help="Directory for the YAML file. The A3M is written under <out-dir>/msa.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.length <= 0:
        raise ValueError("length must be positive.")
    if args.msa_seqs <= 0:
        raise ValueError("--msa-seqs must be positive.")

    name = args.name or str(args.length)
    out_dir: Path = args.out_dir
    msa_dir = out_dir / "msa"
    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    query = random_sequence(args.length, rng)
    yaml_path = out_dir / f"{name}.yaml"
    msa_path = msa_dir / f"{name}.a3m"

    with msa_path.open("w", encoding="utf-8") as handle:
        handle.write(">input\n")
        handle.write(query + "\n")
        for idx in range(args.msa_seqs - 1):
            handle.write(f">{idx}\n")
            handle.write(random_sequence(args.length, rng) + "\n")

    yaml_content = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {query}\n"
        f"      msa: {yaml_msa_path(msa_path)}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Wrote {yaml_path}")
    print(f"Wrote {msa_path} ({args.msa_seqs} sequences, length {args.length})")


if __name__ == "__main__":
    main()
