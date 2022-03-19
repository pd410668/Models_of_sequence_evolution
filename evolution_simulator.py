# !/usr/bin/env python3

import argparse
import sys
from typing import List, Callable
import math
import numpy as np


DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.25

DEFAULT_STEP = 0.01
DEFAULT_MAX_T = 1.0


# Jukes-Cantor model (JC69)
def pJC(a: str, b: str, /,
        t: float, alpha: float = DEFAULT_ALPHA):
    same = 0.25 + 0.75 * math.e ** (-4 * alpha * t)
    replace = 0.25 - 0.25 * math.e ** (-4 * alpha * t)

    same_n = 0
    replace_n = 0
    for a1, a2 in zip(a, b):
        if a1 == "-" or a2 == "-":
            continue
        elif a1 == a2:
            same_n += 1
        else:
            replace_n += 1

    probability = (same ** same_n) * (replace ** replace_n)
    return probability


# Kimura model (K80)
def pK(a: str, b: str, /,
    t: float, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):

    def is_transition(n1: str, n2: str) -> bool:
        return n1 != n2 and ((n1 in {"A", "G"} and n2 in {"A", "G"}) or (n1 in {"C", "T"} and n2 in {"C", "T"}))

    def is_transversion(n1: str, n2: str) -> bool:
        return n1 != n2 and not is_transition(n1, n2)

    same = 0.25 + 0.25 * math.e ** (-4 * beta * t) + 0.5 * math.e ** (-2 * (alpha + beta) * t)
    transition = 0.25 + 0.25 * math.e ** (-4 * beta * t) - 0.5 * math.e ** (-2 * (alpha + beta) * t)
    transversion = 0.25 - 0.25 * math.e ** (-4 * beta * t)

    same_n = 0
    transi_n = 0
    transv_n = 0

    for a1, a2 in zip(a, b):
        if a1 == "-" or a2 == "-":
            continue
        elif a1 == a2:
            same_n += 1
        elif is_transition(a1, a2):
            transi_n += 1
        elif is_transversion(a1, a2):
            transv_n += 1

    probability = (same ** same_n) * (transition ** transi_n) * (transversion ** transv_n)
    return probability


def plot(data: list, step: float, title: str, output_file: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    def format_fn(tick_val, tick_pos):
        return tick_val * step

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=[8, 6])
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.plot(data, linewidth=2)
    plt.title(title, fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    fig.savefig(output_file)


def plot_K(a: str, b: str, /,
           alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
           max_t: float = 1.0, step: float = DEFAULT_STEP, output_file: str = ""):
    probs = [pK(a, b, t, alpha, beta) for t in np.arange(0, max_t, step)]
    tmax = optT_p(probs, step)
    plot(probs, step, title=f"Kimura model: alpha = {alpha}, beta = {beta}, t_max = {tmax}", output_file=output_file)


def plot_JC(a: str, b: str, /,
            alpha: float = DEFAULT_ALPHA, *,
            max_t: float = DEFAULT_MAX_T, step: float = DEFAULT_STEP, output_file: str = ""):
    probs = [pJC(a, b, t, alpha) for t in np.arange(0, max_t, step)]
    tmax = optT_p(probs, step)
    plot(probs, step, title=f"Jukes-Cantor model: alpha = {alpha}, t_max = {tmax}", output_file=output_file)


def optT_p(probs: List[float], step: float) -> float:
    tmax_index = probs.index(max(probs))
    return tmax_index * step


def optT(a: str, b: str, p: Callable, /,
         max_t: float = DEFAULT_MAX_T, step: float = DEFAULT_STEP):
    probs = [p(a, b, t) for t in np.arange(0, max_t, step)]
    return optT_p(probs, step)


def get_two_seqs(filename: str) -> List[str]:
    from Bio import SeqIO
    seqs = [str(record.seq) for record in SeqIO.parse(filename, "fasta")]
    if len(seqs) < 2:
        print("ERROR! Needs at least two sequences.")
        sys.exit(1)
    return seqs[:2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Probability of sequences transforms in evolution model with time")
    parser.add_argument("fasta", type=str, help="fasta file with two aligned sequences")
    parser.add_argument("-m", "--model", type=str, default="jukes-cantor", choices=["kimura", "jukes-cantor"],
                        help="evolution model")
    parser.add_argument("-a", "--alpha", type=float, default=DEFAULT_ALPHA, help="alpha param")
    parser.add_argument("-b", "--beta", type=float, default=DEFAULT_BETA, help="beta param (for Kimura model)")
    parser.add_argument("-t", "--time", type=float, default=DEFAULT_MAX_T, help="max time")
    parser.add_argument("-s", "--step", type=float, default=DEFAULT_STEP, help="step")
    parser.add_argument("-o", "--output", type=str, help="output file")
    args = parser.parse_args()

    output_file = args.output if args.output else "evolution_model.png"
    seqs = get_two_seqs(args.fasta)
    if args.model == "jukes-cantor":
        plot_JC(seqs[0], seqs[1], args.alpha, max_t=args.time, step=args.step, output_file=output_file)
    elif args.model == "kimura":
        plot_K(seqs[0], seqs[1], args.alpha, args.beta, max_t=args.time, step=args.step, output_file=output_file)
    else:
        sys.exit(1)

