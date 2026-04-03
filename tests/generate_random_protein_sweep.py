#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(AA_ALPHABET, k=length))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random single-chain TT-Boltz inputs and matching random A3M files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/random_protein_sweep"),
        help="Output directory for inputs and MSA files.",
    )
    parser.add_argument("--min-len", type=int, default=32, help="Minimum sequence length.")
    parser.add_argument("--max-len", type=int, default=1536, help="Maximum sequence length.")
    parser.add_argument("--step", type=int, default=32, help="Length increment.")
    parser.add_argument(
        "--msa-seqs",
        type=int,
        default=8192,
        help="Total number of sequences per MSA file, including the query.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducibility.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.min_len <= 0 or args.max_len < args.min_len or args.step <= 0:
        raise ValueError("Invalid length range. Require: 0 < min_len <= max_len and step > 0.")
    if args.msa_seqs < 1:
        raise ValueError("--msa-seqs must be >= 1.")

    out_dir: Path = args.out_dir
    inputs_dir = out_dir / "inputs"
    msa_dir = out_dir / "msa"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    lengths = list(range(args.min_len, args.max_len + 1, args.step))
    manifest_lines = ["name\tlength\tmsa_path\tyaml_path"]

    for length in lengths:
        name = f"rand_len_{length:04d}"
        query = random_sequence(length, rng)

        msa_path = msa_dir / f"{name}.a3m"
        with msa_path.open("w", encoding="utf-8") as f:
            f.write(">query\n")
            f.write(query + "\n")
            for idx in range(1, args.msa_seqs):
                f.write(f">seq_{idx:05d}\n")
                f.write(random_sequence(length, rng) + "\n")

        yaml_path = inputs_dir / f"{name}.yaml"
        yaml_content = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {query}\n"
            f"      msa: {msa_path.resolve().as_posix()}\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        manifest_lines.append(f"{name}\t{length}\t{msa_path}\t{yaml_path}")

    manifest_path = out_dir / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(
        f"Generated {len(lengths)} inputs in {out_dir} "
        f"(lengths {args.min_len}..{args.max_len} step {args.step}, msa_seqs={args.msa_seqs})."
    )


if __name__ == "__main__":
    main()

