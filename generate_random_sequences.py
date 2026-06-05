#!/usr/bin/env python3
"""
generate_random_sequences.py

Generate a control corpus of random protein sequences for training
dark-whiteGPLM. Residues are drawn i.i.d. either uniformly over the 20 standard
amino acids (--composition uniform) or according to the empirical amino-acid
frequencies of the algal proteome (--composition algae). Sequence lengths are
sampled from an empirical length distribution (default: prompts_1000algae.csv)
in both modes, so the only difference between the two corpora is residue
composition.

The two modes give two different floors:
  * uniform : maximum entropy; the model can learn nothing, so cross-entropy
              converges to ~ln(20) ~= 3.0 nats. Pure floor.
  * algae   : composition-matched; the model can still learn the single-residue
              (unigram) bias but no higher-order structure, so the floor sits
              slightly lower. Isolates ordering/structure from composition.
Generating both lets you test whether the real datasets' loss separates from
composition alone.

Purpose: random sequences carry no learnable structure, so a model trained on
them establishes a floor for LLM training loss -- the cross-entropy reference
against which the dark / annotated / algae loss curves are interpreted (cf. the
training-dynamics figure in the ELF-NET supplementary report).

Output format matches prompts_1000algae.csv: one sequence per line, each
terminated with '>' (the separator used during training). This is NOT
conventional FASTA (no '>' headers, no line wrapping).

Examples:
    # 1000 sequences, fixed seed for reproducibility
    python generate_random_sequences.py -n 1000 --seed 42

    # Large corpus matched in size to the algae dataset, custom output path
    python generate_random_sequences.py -n 400000 --seed 1337 -o data/char_random/input_prompts.csv

    # Draw lengths from a different reference file
    python generate_random_sequences.py -n 1000 --seed 1 --length_ref data/char_dark/input.txt
"""

import argparse
import os
import random

# The 20 standard amino acids (the model's vocab also includes 'X', '<', '>',
# and newline, but those are not part of the random residue alphabet).
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def load_empirical_lengths(path, sep=">"):
    """Return the list of sequence lengths found in a prompts-style file.

    Each non-empty line is treated as one sequence; a trailing separator
    character (default '>') is stripped before measuring length.
    """
    lengths = []
    with open(path) as fh:
        for line in fh:
            seq = line.strip().rstrip(sep)
            if seq:
                lengths.append(len(seq))
    if not lengths:
        raise ValueError(f"No sequences found in length reference: {path}")
    return lengths


def load_composition_weights(path, alphabet):
    """Return per-residue weights from a file's empirical amino-acid frequencies.

    Counts only characters in `alphabet` (non-standard symbols such as 'X' are
    ignored). Weights are aligned positionally with `alphabet`.
    """
    counts = {a: 0 for a in alphabet}
    with open(path) as fh:
        for line in fh:
            for ch in line.strip().rstrip(">"):
                if ch in counts:
                    counts[ch] += 1
    weights = [counts[a] for a in alphabet]
    if sum(weights) == 0:
        raise ValueError(f"No alphabet residues found in composition reference: {path}")
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Generate random protein sequences (max-entropy training-loss floor).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--num_sequences", type=int, default=1000,
                        help="Number of random sequences to generate.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility.")
    parser.add_argument("-c", "--composition", choices=["uniform", "algae"], default="uniform",
                        help="Residue distribution: 'uniform' (max-entropy floor) or "
                             "'algae' (match empirical amino-acid frequencies of --composition_ref).")
    parser.add_argument("--composition_ref", default="prompts_1000algae.csv",
                        help="File to derive empirical amino-acid frequencies from "
                             "when --composition algae.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: random_<n>seqs_seed<seed>.csv).")
    parser.add_argument("--length_ref", default="prompts_1000algae.csv",
                        help="File to draw the empirical length distribution from "
                             "(one sequence per line, trailing '>' optional).")
    parser.add_argument("--alphabet", default=AA20,
                        help="Residue alphabet to sample uniformly from.")
    parser.add_argument("--min_len", type=int, default=None,
                        help="Discard reference lengths below this value (optional).")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Discard reference lengths above this value (optional).")
    args = parser.parse_args()

    if args.num_sequences < 1:
        parser.error("--num_sequences must be >= 1")

    lengths_pool = load_empirical_lengths(args.length_ref)
    if args.min_len is not None:
        lengths_pool = [n for n in lengths_pool if n >= args.min_len]
    if args.max_len is not None:
        lengths_pool = [n for n in lengths_pool if n <= args.max_len]
    if not lengths_pool:
        parser.error("No reference lengths remain after applying --min_len/--max_len.")

    alphabet = args.alphabet
    if args.composition == "algae":
        weights = load_composition_weights(args.composition_ref, alphabet)
    else:
        weights = None  # equal weights -> uniform

    output = args.output or f"random_{args.composition}_{args.num_sequences}seqs_seed{args.seed}.csv"

    rng = random.Random(args.seed)
    total_residues = 0
    with open(output, "w") as fh:
        for _ in range(args.num_sequences):
            length = rng.choice(lengths_pool)
            seq = "".join(rng.choices(alphabet, weights=weights, k=length))
            fh.write(seq + ">\n")
            total_residues += length

    print(f"Wrote {args.num_sequences} random sequences to {output}")
    if args.composition == "algae":
        comp_desc = f"algae frequencies from {args.composition_ref}"
    else:
        comp_desc = "uniform (equal probability)"
    print(f"  seed={args.seed}, composition={args.composition} -> {comp_desc}")
    print(f"  alphabet={alphabet} ({len(alphabet)} residues)")
    print(f"  length distribution sampled from {args.length_ref} "
          f"(pool size {len(lengths_pool)})")
    print(f"  total residues: {total_residues:,} "
          f"(mean length {total_residues / args.num_sequences:.1f})")


if __name__ == "__main__":
    main()
