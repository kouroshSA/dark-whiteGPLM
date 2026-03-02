#!/usr/bin/env python3
"""
Batch inference for nanoGPT protein language models.

Reads prompts from a CSV file, runs each through a trained nanoGPT/GPT-2
checkpoint, and writes a two-column CSV with the prompt and generated output.

Includes six mitigations for GPT-2 repetitive degeneration:
    1. Repetition penalty (Keskar et al., CTRL 2019) — penalizes logits of
       tokens already present in the generated sequence.
    2. Frequency penalty — linearly penalizes tokens proportional to how many
       times they have appeared (stronger than binary repetition penalty).
    3. Top-p / nucleus sampling (Holtzman et al., 2020) — dynamically selects
       the smallest set of tokens whose cumulative probability exceeds p.
    4. N-gram blocking — prevents any n-gram from appearing more than once
       in the generated output.
    5. Max consecutive repeats — caps the number of times the same token can
       appear in a row (prevents KKKKK, EEEEE, AAAAA runs).
    6. Max motif repeat — caps how many times a di/tri-peptide motif can
       repeat consecutively (prevents AEAEAE, GKEGKEGKE patterns).

Requirements:
    - A trained checkpoint (ckpt.pt) in the model directory
    - A meta.pkl vocabulary file in data/<dataset>/
    - model.py and configurator.py in the working directory

Usage:
    # Basic — defaults include anti-repetition controls for cleaner output
    python batch_inference.py -i prompts.csv -n 100 -t 0.8 -k 20

    # Disable all anti-repetition (raw model output)
    python batch_inference.py -i prompts.csv -n 100 --rep_penalty 1.0 \
        --freq_penalty 0.0 --no_repeat_ngram 0 --max_consecutive 0 \
        --max_motif_repeat 0

    # Aggressive anti-repetition for long generation
    python batch_inference.py -i prompts.csv -n 200 -t 0.9 -k 40 \
        --rep_penalty 1.3 --freq_penalty 1.0 --no_repeat_ngram 4 --max_consecutive 2

    # With nucleus sampling
    python batch_inference.py -i prompts.csv -n 100 -t 0.8 --top_p 0.95

    # Specify model directory and output file
    python batch_inference.py -i prompts.csv -m out_dark_40ki_standard -o results.csv

    # Use CPU
    python batch_inference.py -i prompts.csv --device cpu

Input format (prompt.csv):
    One prompt per line. If the file has a header row, use --header to skip it.
    Lines are stripped of whitespace; empty lines are skipped.

Output format:
    CSV with two columns: prompt, output
    If the generated output contains '>' or '<', it is trimmed at that character.

Author: Kourosh Salehi-Ashtiani
Based on nanoGPT (Karpathy) sampling infrastructure.
"""

import os
import sys
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch inference for nanoGPT protein language models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python batch_inference.py -i prompts.csv -n 100 -t 0.8 -k 20
  python batch_inference.py -i prompts.csv -n 200 -t 0.9 -k 40 --freq_penalty 1.0 --max_consecutive 2
  python batch_inference.py -i prompts.csv -n 100 --rep_penalty 1.0 --freq_penalty 0 --no_repeat_ngram 0 --max_consecutive 0
  python batch_inference.py -i prompts.csv -m out_dark_40ki_standard -o results.csv --device cpu
        """,
    )
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to CSV file containing prompts (one per line)')
    parser.add_argument('-o', '--output_file', type=str, default=None,
                        help='Path for output CSV (default: <input_stem>_output.csv)')
    parser.add_argument('-m', '--model_dir', type=str, default='out',
                        help='Directory containing ckpt.pt (default: out)')
    parser.add_argument('-n', '--max_new_tokens', type=int, default=1,
                        help='Number of tokens to generate per prompt (default: 1)')
    parser.add_argument('-t', '--temperature', type=float, default=0.1,
                        help='Sampling temperature (default: 0.1)')
    parser.add_argument('-k', '--top_k', type=int, default=2,
                        help='Top-k sampling; 0 to disable (default: 2)')
    parser.add_argument('--top_p', type=float, default=0.0,
                        help='Top-p / nucleus sampling threshold; 0 to disable (default: 0)')
    parser.add_argument('--rep_penalty', type=float, default=1.2,
                        help='Repetition penalty factor; 1.0 = off, >1.0 = penalize repeats '
                             '(default: 1.2)')
    parser.add_argument('--freq_penalty', type=float, default=0.5,
                        help='Frequency penalty: subtract freq_penalty * count(token) from logits; '
                             '0 = off (default: 0.5)')
    parser.add_argument('--no_repeat_ngram', type=int, default=3,
                        help='Block repeated n-grams of this size; 0 = off (default: 3)')
    parser.add_argument('--max_consecutive', type=int, default=3,
                        help='Max times the same token can appear consecutively; '
                             '0 = unlimited (default: 3)')
    parser.add_argument('--max_motif_repeat', type=int, default=2,
                        help='Max times a di/tri-peptide motif can repeat consecutively; '
                             'e.g. 2 allows AEAE but blocks AEAEAE; 0 = unlimited (default: 2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on: cuda or cpu (default: cuda)')
    parser.add_argument('--header', action='store_true',
                        help='Skip the first line of the input CSV as a header')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    return parser.parse_args()


def generate_with_penalties(model, idx, max_new_tokens, temperature, top_k,
                            top_p, rep_penalty, freq_penalty, no_repeat_ngram,
                            max_consecutive, max_motif_repeat, block_size):
    """
    Autoregressive generation with repetition mitigation.

    Unlike model.generate(), this function applies:
      - Repetition penalty: scales down logits of previously seen tokens
      - Frequency penalty: linearly penalizes tokens by occurrence count
      - N-gram blocking: prevents exact n-gram repetition
      - Max consecutive: caps runs of the same token
      - Max motif repeat: caps repeating di/tri-peptide motifs
      - Top-p (nucleus) sampling: dynamic vocabulary truncation

    Args:
        model:              GPT model in eval mode
        idx:                (1, T) LongTensor of prompt token ids
        max_new_tokens:     number of new tokens to generate
        temperature:        sampling temperature (>0)
        top_k:              keep only top-k logits; 0 = no top-k filtering
        top_p:              keep smallest set with cumulative prob >= top_p; 0 = off
        rep_penalty:        multiplicative penalty for repeated tokens; 1.0 = off
        freq_penalty:       linear frequency penalty factor; 0.0 = off
        no_repeat_ngram:    block n-grams of this size from repeating; 0 = off
        max_consecutive:    max consecutive repeats of the same token; 0 = unlimited
        max_motif_repeat:   max consecutive repeats of di/tri-peptide motifs; 0 = off
        block_size:         model's maximum context length

    Returns:
        (1, T + max_new_tokens) LongTensor of token ids
    """
    import torch
    import torch.nn.functional as F
    from collections import Counter

    prompt_len = idx.size(1)

    for step in range(max_new_tokens):
        # Crop context to block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        generated_ids = idx[0, prompt_len:].tolist()

        # ── Repetition penalty (Keskar et al., CTRL 2019) ─────────
        if rep_penalty != 1.0 and generated_ids:
            unique_ids = list(set(generated_ids))
            penalty_logits = logits[0, unique_ids]
            # If logit > 0, divide by penalty (make less likely)
            # If logit < 0, multiply by penalty (make even less likely)
            penalty_logits = torch.where(
                penalty_logits > 0,
                penalty_logits / rep_penalty,
                penalty_logits * rep_penalty,
            )
            logits[0, unique_ids] = penalty_logits

        # ── Frequency penalty (linear, proportional to count) ────
        if freq_penalty > 0.0 and generated_ids:
            token_counts = Counter(generated_ids)
            for token_id, count in token_counts.items():
                logits[0, token_id] -= freq_penalty * count

        # ── Max consecutive repeats ──────────────────────────────
        if max_consecutive > 0 and len(generated_ids) >= max_consecutive:
            # Check if the last max_consecutive tokens are all the same
            tail = generated_ids[-max_consecutive:]
            if len(set(tail)) == 1:
                # Ban this token from being generated again
                logits[0, tail[0]] = -float('Inf')

        # ── Max motif repeat (di/tri-peptide patterns) ───────────
        if max_motif_repeat > 0 and len(generated_ids) >= 4:
            for motif_len in (2, 3):
                window_size = motif_len * max_motif_repeat
                if len(generated_ids) >= window_size:
                    tail = generated_ids[-window_size:]
                    motif = tail[:motif_len]
                    # Check if the entire window is a repeat of this motif
                    is_repeat = all(
                        tail[i] == motif[i % motif_len]
                        for i in range(window_size)
                    )
                    if is_repeat:
                        # Ban the token that would start the next repeat
                        logits[0, motif[0]] = -float('Inf')

        # ── N-gram blocking ───────────────────────────────────────
        if no_repeat_ngram > 1 and idx.size(1) >= no_repeat_ngram:
            all_tokens = idx[0].tolist()
            # Find n-grams that have already appeared
            banned_tokens = set()
            ngram_prefix = tuple(all_tokens[-(no_repeat_ngram - 1):])
            for i in range(len(all_tokens) - no_repeat_ngram + 1):
                window = tuple(all_tokens[i:i + no_repeat_ngram - 1])
                if window == ngram_prefix:
                    # The token that followed this prefix before is banned
                    banned_tokens.add(all_tokens[i + no_repeat_ngram - 1])
            if banned_tokens:
                logits[0, list(banned_tokens)] = -float('Inf')

        # ── Temperature scaling ───────────────────────────────────
        logits = logits / temperature

        # ── Top-k filtering ───────────────────────────────────────
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # ── Top-p / nucleus filtering ─────────────────────────────
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            # Shift right so the first token above threshold is kept
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_indices_to_remove] = -float('Inf')
            # Scatter back to original ordering
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # ── Sample ────────────────────────────────────────────────
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    args = parse_args()

    # Deferred imports — keeps -h fast and dependency-free
    import pickle
    from contextlib import nullcontext
    import torch
    import torch.nn.functional as F
    from model import GPTConfig, GPT

    # Reset sys.argv so configurator.py (exec'd by nanoGPT internals) sees nothing
    sys.argv = [sys.argv[0]]

    # Derive output path if not specified
    if args.output_file is None:
        stem = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = stem + '_output.csv'

    # ── Seed ──────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ── Device and precision ──────────────────────────────────────────
    device = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    if device_type == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        device = 'cpu'
        device_type = 'cpu'

    if device_type == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dtype = 'float32'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # ── Load model ────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        sys.exit(f'Error: checkpoint not found at {ckpt_path}')

    print(f'Loading model from {ckpt_path}...')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    block_size = checkpoint['model_args'].get('block_size', 1024)
    model.eval()
    model.to(device)

    # ── Load vocabulary ───────────────────────────────────────────────
    dataset_name = checkpoint['config']['dataset']
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    if not os.path.exists(meta_path):
        sys.exit(f'Error: vocabulary file not found at {meta_path}')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    # ── Read prompts ──────────────────────────────────────────────────
    if not os.path.exists(args.input_file):
        sys.exit(f'Error: input file not found: {args.input_file}')

    prompts = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if args.header and row_idx == 0:
                continue
            line = row[0].strip() if row else ''
            if line:
                prompts.append(line)

    if not prompts:
        sys.exit('Error: no prompts found in input file')

    # Check if any anti-repetition measures are active
    has_penalties = (args.rep_penalty != 1.0 or args.freq_penalty > 0.0
                     or args.no_repeat_ngram > 0 or args.max_consecutive > 0
                     or args.max_motif_repeat > 0 or args.top_p > 0.0)

    print(f'Loaded {len(prompts)} prompts from {args.input_file}')
    print(f'Generation: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, '
          f'top_k={args.top_k}')
    if has_penalties:
        print(f'Anti-repetition: rep_penalty={args.rep_penalty}, '
              f'freq_penalty={args.freq_penalty}, '
              f'no_repeat_ngram={args.no_repeat_ngram}, '
              f'max_consecutive={args.max_consecutive}, '
              f'max_motif_repeat={args.max_motif_repeat}, '
              f'top_p={args.top_p}')
    print(f'Model: {args.model_dir}  |  Output: {args.output_file}')
    print()

    # ── Run inference ─────────────────────────────────────────────────
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'output'])

        with torch.no_grad():
            with ctx:
                for k, prompt in enumerate(prompts):
                    start_ids = encode(prompt)
                    if len(start_ids) > block_size:
                        start_ids = start_ids[-block_size:]
                    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

                    if has_penalties:
                        y = generate_with_penalties(
                            model, x,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            rep_penalty=args.rep_penalty,
                            freq_penalty=args.freq_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            max_consecutive=args.max_consecutive,
                            max_motif_repeat=args.max_motif_repeat,
                            block_size=block_size,
                        )
                    else:
                        y = model.generate(
                            idx=x,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                        )

                    generated = decode(y[0].tolist())

                    # Remove the prompt prefix to get only the new tokens
                    prompt_decoded = decode(start_ids)
                    if generated.startswith(prompt_decoded):
                        output_text = generated[len(prompt_decoded):]
                    else:
                        output_text = generated

                    # Trim at '>' or '<' character if present
                    for trim_char in ('>', '<'):
                        pos = output_text.find(trim_char)
                        if pos != -1:
                            output_text = output_text[:pos]

                    # Remove newlines and surrounding whitespace
                    output_text = output_text.strip().replace('\n', '').replace('\r', '')

                    writer.writerow([prompt, output_text])

                    print(f'  [{k + 1}/{len(prompts)}] ...{prompt[-30:]} -> {output_text}')

    print(f'\nDone. Results saved to {args.output_file}')


if __name__ == '__main__':
    main()
