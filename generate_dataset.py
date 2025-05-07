"""Generate symbolic‑math datasets (train / val / test)."""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────
import argparse, datetime, json, math, os, random, sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── third‑party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from num2words import num2words as _n2w
from tqdm.auto import tqdm
import yaml
from math import gcd as _gcd

# ── helpers ───────────────────────────────────────────────────────────
def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# ══════════════════════════════════════════════════════════════════════
# 1. argument parsing
# ══════════════════════════════════════════════════════════════════════
def parse_args(cli_args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI + YAML defaults."""
    # preliminary parser to locate YAML file
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="generate_dataset_config.yaml")
    cfg, remaining = pre.parse_known_args(cli_args)

    with open(cfg.config) as f:
        defaults = yaml.safe_load(f)

    p = argparse.ArgumentParser(description="Generate dataset")
    p.add_argument("--config", default="generate_dataset_config.yaml")

    # path config
    p.add_argument("--curr_dir", default=defaults.get("curr_dir"))

    # data generation
    p.add_argument("--possible_problems", nargs="+",
                   default=defaults.get("possible_problems"))
    p.add_argument("--complexity", type=int, default=defaults.get("complexity"))
    p.add_argument("--n_samples_dataset", type=int,
                   default=defaults.get("n_samples_dataset"))

    p.add_argument("--train_data_rounds", type=int,
                   default=defaults.get("train_data_rounds"))
    p.add_argument("--val_data_rounds", type=int,
                   default=defaults.get("val_data_rounds"))
    p.add_argument("--test_data_rounds", type=int,
                   default=defaults.get("test_data_rounds"))

    p.add_argument("--string_nums", type=str2bool,
                   default=defaults.get("string_nums"))
    p.add_argument("--limit_solution_digits", type=str2bool,
                   default=defaults.get("limit_solution_digits"))
    p.add_argument("--modify_question_format", type=str2bool,
                   default=defaults.get("modify_question_format"))

    p.add_argument("--seed", type=int, default=defaults.get("seed"))
    return p.parse_args(remaining if cli_args is None else cli_args)

# ══════════════════════════════════════════════════════════════════════
# 2. problem metadata & solvers
# ══════════════════════════════════════════════════════════════════════
_COMMUTATIVE = {
    "addition", "multiplication", "gcd", "lcm",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_nand", "bitwise_nor", "bitwise_nxor",
}
_Q = {
    "addition"       : "What is {x} plus {y}?",
    "multiplication" : "What is {x} times {y}{m}?",
    "division"       : "What is {x} // {y}?",
    "modulo"         : "What is {x} mod {y}?",
    "gcd"            : "What is the GCD of {x} and {y}?",
    "lcm"            : "What is the LCM of {x} and {y}{m}?",
    "square_mod"     : "What is {x}^2 mod {y}?",
    "bitwise_and"    : "What is {x} AND {y}?",
    "bitwise_or"     : "What is {x} OR {y}?",
    "bitwise_xor"    : "What is {x} XOR {y}?",
    "bitwise_nor"    : "What is {x} NOR {y}?",
    "bitwise_nand"   : "What is {x} NAND {y}?",
    "bitwise_nxor"   : "What is {x} NXOR {y}?",
}
def _solve(ptype: str, x: np.ndarray, y: np.ndarray, mod: int) -> np.ndarray:
    if ptype == "addition":         return (x + y).astype(str)
    if ptype == "multiplication":   return ((x * y) % mod).astype(str)
    if ptype == "division":         return (x // y).astype(str)
    if ptype == "modulo":           return (x % y).astype(str)
    if ptype == "gcd":              return np.frompyfunc(_gcd, 2, 1)(x, y).astype(str)
    if ptype == "lcm":              return ((x * y // np.frompyfunc(_gcd, 2, 1)(x, y)) % mod).astype(str)
    if ptype == "square_mod":       return np.mod(np.power(x, 2), y).astype(str)
    if ptype == "bitwise_and":      return (x & y).astype(str)
    if ptype == "bitwise_or":       return (x | y).astype(str)
    if ptype == "bitwise_xor":      return (x ^ y).astype(str)
    if ptype == "bitwise_nor":      return ((~(x | y)) & 0xFFFF).astype(str)
    if ptype == "bitwise_nand":     return ((~(x & y)) & 0xFFFF).astype(str)
    if ptype == "bitwise_nxor":     return ((~(x ^ y)) & 0xFFFF).astype(str)
    raise ValueError(f"unknown problem_type {ptype}")

_FMT_POOL = {
    # ────────────── ADDITION ──────────────
    "addition": [
        "{x} + {y}",
        "{x}+{y}",
        "Add {x} and {y}.",
        "Work out {x} + {y}.",
        "Sum {x} and {y}.",
        "Total of {x} and {y}.",
        "Add together {x} and {y}.",
        "What is {x} plus {y}?",
        "What is the sum of {x} and {y}?",
        "Calculate {x} + {y}.",
    ],

    # ── Multiplication - superset of Google's mathematics dataset templates (arithmetic module) ───────────────
    "multiplication": [
        # bare expressions
        "{x}*{y}{m}",
        "{x} * {y}{m}",

        # imperative prompts
        "Calculate {x}*{y}{m}.",
        "Calculate {x} * {y}{m}.",
        "Work out {x}*{y}{m}.",
        "Work out {x} * {y}{m}.",
        "Work out {x} times {y}{m}.",
        "Multiply {x} and {y}{m}.",

        # noun‑phrase prompts
        "Product of {x} and {y}{m}.",
        "What is the product of {x} and {y}{m}?",

        # everyday “times” phrasing
        "{x} times {y}{m}",
        "What is {x} times {y}{m}?",

        # explicit descriptive form
        "Find the result of {x} multiplied by {y}{m}.",
    ],

    # ───────────── DIVISION ─────────────
    "division": [
        "{x}//{y}",
        "{x} // {y}",
        "Divide {x} by {y}.",
        "{x} divided by {y}",
        "What is {x} divided by {y}?",
        "Calculate {x} divided by {y}.",
        "Compute {x} over {y}.",
    ],

    # ─────────────── MODULO ───────────────
    "modulo": [
        "{x} mod {y}",
        "{x}%{y}",
        "{x} % {y}",
        "Find {x} mod {y}.",
        "What is {x} mod {y}?",
        "Calculate {x} modulo {y}.",
        "Compute {x} mod {y}.",
    ],

    # ──────────────── GCD ────────────────
    "gcd": [
        "gcd({x}, {y})",
        "GCD({x}, {y})",
        "What is the GCD of {x} and {y}?",
        "Calculate the greatest common divisor of {x} and {y}.",
        "Find gcd of {x} and {y}.",
        "Compute GCD({x}, {y}).",
    ],

    # ──────────────── LCM ────────────────
    "lcm": [
        "Find lcm({x}, {y}){m}.",
        "What is the least common multiple of {x} and {y}{m}?",
        "Calculate LCM({x}, {y}){m}.",
        "LCM({x}, {y}){m}",
        "Compute the least common multiple of {x} and {y}{m}.",
    ],

    # ─────────── SQUARE MOD ─────────────
    "square_mod": [
        "{x}^2 mod {y}",
        "({x}^2) mod {y}",
        "What is {x} squared mod {y}?",
        "Calculate {x}^2 mod {y}.",
        "Compute {x} squared modulo {y}.",
    ],

    # ─────────── BITWISE AND ────────────
    "bitwise_and": [
        "{x} & {y}",
        "{x}&{y}",
        "{x} AND {y}",
        "Calculate {x} AND {y}.",
        "Compute the bitwise AND of {x} and {y}.",
        "What is {x} AND {y}?",
    ],

    # ─────────── BITWISE OR ─────────────
    "bitwise_or": [
        "{x} | {y}",
        "{x}|{y}",
        "{x} OR {y}",
        "Calculate {x} OR {y}.",
        "Compute the bitwise OR of {x} and {y}.",
        "What is {x} OR {y}?",
    ],

    # ─────────── BITWISE XOR ────────────
    "bitwise_xor": [
        "{x} ^ {y}",
        "{x}^{y}",
        "{x} XOR {y}",
        "Calculate {x} XOR {y}.",
        "Compute the bitwise XOR of {x} and {y}.",
        "What is {x} XOR {y}?",
    ],

    # ─────────── BITWISE NOR ────────────
    "bitwise_nor": [
        "{x} NOR {y}",
        "Bitwise NOR of {x} and {y}.",
        "Calculate {x} NOR {y}.",
        "Compute the bitwise NOR of {x} and {y}.",
        "What is {x} NOR {y}?",
    ],

    # ─────────── BITWISE NAND ───────────
    "bitwise_nand": [
        "{x} NAND {y}",
        "Bitwise NAND of {x} and {y}.",
        "Calculate {x} NAND {y}.",
        "Compute the bitwise NAND of {x} and {y}.",
        "What is {x} NAND {y}?",
    ],

    # ─────────── BITWISE NXOR ───────────
    "bitwise_nxor": [
        "{x} NXOR {y}",
        "{x} XNOR {y}",
        "Bitwise NXOR of {x} and {y}.",
        "Calculate {x} NXOR {y}.",
        "Compute the bitwise NXOR of {x} and {y}.",
        "What is {x} NXOR {y}?",
    ],
}

def _random_format(ptype: str, rng: np.random.Generator) -> str:
    pool = _FMT_POOL.get(ptype)
    return rng.choice(pool) if pool is not None else _Q[ptype]

# ══════════════════════════════════════════════════════════════════════
# 3. dataset builder
# ══════════════════════════════════════════════════════════════════════
def build_dataset(
    samples: int,
    *,
    complexity: int,
    possible_problems: List[str],
    string_nums: bool,
    limit_solution_digits: bool,
    modify_question_format: bool,
    seed: int,
) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    mod   = 10 ** (complexity + 1)
    n_max = mod - 1

    allowed = possible_problems
    half  = n_max * (n_max - 1) // 2
    equal = n_max
    cap   = sum((half + equal) if p in _COMMUTATIVE else half for p in allowed)
    if samples > cap:
        raise ValueError(f"Requested {samples:,} rows but only {cap:,} unique "

                         f"questions possible at this complexity.")

    seen: set[Tuple[str, int, int]] = set()
    rows  = []
    with tqdm(total=samples, unit="row") as pbar:
        while len(rows) < samples:
            k = min(500_000, samples - len(rows))
            ptypes = rng.choice(allowed, size=k)
            x      = rng.integers(1, n_max + 1, size=k)
            y      = rng.integers(1, n_max + 1, size=k)

            non_comm = ~np.isin(ptypes, list(_COMMUTATIVE))
            swap     = non_comm & (x <= y)
            x[swap], y[swap] = y[swap], x[swap]

            equal_mask = non_comm & (x == y)
            while equal_mask.any():
                y[equal_mask] = rng.integers(1, n_max + 1, size=equal_mask.sum())
                swap = equal_mask & (x <= y)
                x[swap], y[swap] = y[swap], x[swap]
                equal_mask = non_comm & (x == y)

            sol_str = np.empty(k, dtype=object)
            for p in np.unique(ptypes):
                mask = ptypes == p
                sol_str[mask] = _solve(p, x[mask], y[mask], mod)

            for i in range(k):
                p = ptypes[i]
                k1, k2 = (sorted((int(x[i]), int(y[i]))) if p in _COMMUTATIVE
                          else (int(x[i]), int(y[i])))
                if (p, k1, k2) in seen:
                    continue
                seen.add((p, k1, k2))

                a, b = (_n2w(x[i]), _n2w(y[i])) if string_nums else (x[i], y[i])
                sol  = _n2w(int(sol_str[i])) if string_nums else sol_str[i]

                fmt  = _random_format(p, rng) if modify_question_format else _Q[p]
                mstr = f" mod {mod}" if (p in {"multiplication", "lcm"} and
                                         limit_solution_digits) else ""
                quest = fmt.replace("{m}", mstr).format(x=a, y=b)
                rows.append((p, x[i], y[i], quest, sol))
                pbar.update(1)
                if len(rows) == samples:
                    break
    return pd.DataFrame(rows,
                        columns=["problem_type", "x", "y", "question", "solution"])

# ══════════════════════════════════════════════════════════════════════
# 4. dataset split & IO utilities
# ══════════════════════════════════════════════════════════════════════
def split_dataset(df: pd.DataFrame,
                  train_n: int, val_n: int, test_n: int,
                  seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(df.index.to_numpy())
    total_needed = train_n + val_n + test_n
    if total_needed > len(perm):
        raise ValueError("Not enough rows to satisfy requested split sizes.")

    train_idx = perm[:train_n]
    val_idx   = perm[train_n:train_n + val_n]
    test_idx  = perm[train_n + val_n : train_n + val_n + test_n]

    return (df.loc[train_idx].reset_index(drop=True),
            df.loc[val_idx  ].reset_index(drop=True),
            df.loc[test_idx ].reset_index(drop=True))

def build_dataset_paths(args) -> Tuple[Path, Path, Path, Path]:
    base = args.curr_dir + "/datasets/"
    name = f"symbolic_math_dataset_{args.complexity}_complexity"
    if args.string_nums:
        name += "_string_representation"
    if not args.limit_solution_digits:
        name += "_unlimited_solution_digits"
    if args.modify_question_format:
        name += "_random_question_format"

    all_path   = base + f"{name}_{args.n_samples_dataset}_samples.csv"
    train_path = base + f"{name}_{args.train_data_rounds}_training_samples.csv"
    val_path   = base + f"{name}_{args.val_data_rounds  }_validation_samples.csv"
    test_path  = base + f"{name}_{args.test_data_rounds }_testing_samples.csv"
    return all_path, train_path, val_path, test_path

def save_if_missing(path: Path, df: pd.DataFrame) -> None:
    if os.path.exists(path):
        print(f"Dataset {path} already exists, skipping...")
        return
    print("Saving dataset:", path)
    df.to_csv(path, index=False)

# ══════════════════════════════════════════════════════════════════════
# 5. main entry‑point
# ══════════════════════════════════════════════════════════════════════
def main(cli_args: Optional[List[str]] = None) -> None:
    args = parse_args(cli_args)

    args.curr_dir = str(Path(args.curr_dir).expanduser())

    df = build_dataset(
        samples=args.n_samples_dataset,
        complexity=args.complexity,
        possible_problems=args.possible_problems,
        string_nums=args.string_nums,
        limit_solution_digits=args.limit_solution_digits,
        modify_question_format=args.modify_question_format,
        seed=args.seed,
    )

    df_train, df_val, df_test = split_dataset(
        df, args.train_data_rounds, args.val_data_rounds,
        args.test_data_rounds, args.seed)

    all_p, train_p, val_p, test_p = build_dataset_paths(args)

    if not os.path.exists(f"{args.curr_dir}/datasets"):
        print("Creating directory:", f"{args.curr_dir}/datasets")
        os.mkdir(f"{args.curr_dir}/datasets")

    save_if_missing(all_p,   df)
    save_if_missing(train_p, df_train)
    save_if_missing(val_p,   df_val)
    save_if_missing(test_p,  df_test)

# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
