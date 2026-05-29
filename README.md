# dark-whiteGPLM

Character-level protein language models trained on dark and annotated (white) proteome datasets from the TARA-Oceans marine microalgal survey. Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Overview

To be provided

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/dark-whiteGPLM.git
cd dark-whiteGPLM

# Option 1: Create a conda environment (recommended)
conda create -n gpt python=3.10
conda activate gpt
pip install -r requirements.txt

# Option 2: Install directly with pip
pip install -r requirements.txt
```

### Verify installation

```bash
python batch_inference.py -h
```

## Repository Structure

```
dark-whiteGPLM/
|-- model.py                 # GPT model definition (~300 lines)
|-- train_.py                # Training loop
|-- batch_inference.py       # Batch inference with anti-repetition controls
|-- configurator.py          # Configuration utility for training
|-- config/
|   |-- train_par_gpt2-s_scratch.py   # Training config (GPT-2 small, from scratch)
|   +-- finetune_label3.py            # Fine-tuning config
|-- data/
|   |-- char_dark/           # Dark proteome dataset
|   |   |-- prepare.py       # Character-level tokenizer
|   |   |-- meta.pkl         # Vocabulary (stoi/itos mappings)
|   |   +-- input.txt        # Raw protein sequences (FASTA without headers)
|   |-- char_white/          # Annotated proteome dataset
|   +-- char_algae/          # Combined algae dataset
|-- out/                     # Default model checkpoint directory
|   +-- ckpt.pt              # Trained model checkpoint
|-- prompts_1000algae.csv    # Example prompts (1000 algal protein sequences)
|-- requirements.txt
|-- LICENSE
+-- README.md
```

## Usage

### Batch Inference

Generate protein sequences from a trained model using prompts:

```bash
# Basic usage with default anti-repetition controls
python batch_inference.py -i prompts_1000algae.csv -n 100 -t 0.8 -k 20

# Specify a different model directory
python batch_inference.py -i prompts_1000algae.csv -n 100 -t 0.8 -k 20 \
    -m out_dark_40ki_standard

# Aggressive anti-repetition for longer sequences
python batch_inference.py -i prompts_1000algae.csv -n 200 -t 0.9 -k 40 \
    --rep_penalty 1.3 --freq_penalty 1.0 --no_repeat_ngram 4 --max_consecutive 2

# With nucleus sampling
python batch_inference.py -i prompts_1000algae.csv -n 100 -t 0.8 --top_p 0.95

# Run on CPU
python batch_inference.py -i prompts_1000algae.csv -n 100 -t 0.8 -k 20 --device cpu

# Disable all anti-repetition controls (raw model output)
python batch_inference.py -i prompts_1000algae.csv -n 100 -t 0.8 -k 20 \
    --rep_penalty 1.0 --freq_penalty 0 --no_repeat_ngram 0 \
    --max_consecutive 0 --max_motif_repeat 0
```

#### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i` | (required) | Input CSV file with prompts (one per line) |
| `-o` | `<input>_output.csv` | Output CSV file path |
| `-m` | `out` | Model directory containing `ckpt.pt` |
| `-n` | `1` | Number of tokens to generate per prompt |
| `-t` | `0.1` | Sampling temperature |
| `-k` | `2` | Top-k sampling (0 to disable) |
| `--top_p` | `0.0` | Top-p / nucleus sampling threshold (0 to disable) |
| `--rep_penalty` | `1.2` | Repetition penalty factor (1.0 = off) |
| `--freq_penalty` | `0.5` | Frequency penalty: subtracts `value * count(token)` from logits (0 = off) |
| `--no_repeat_ngram` | `3` | Block repeated n-grams of this size (0 = off) |
| `--max_consecutive` | `3` | Max consecutive repeats of the same amino acid (0 = unlimited) |
| `--max_motif_repeat` | `2` | Max repeats of di/tri-peptide motifs, e.g. 2 allows AEAE but blocks AEAEAE (0 = unlimited) |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--header` | `false` | Skip first line of input CSV as header |
| `--seed` | random | Random seed for reproducibility |

#### Anti-Repetition Controls

GPT-2 autoregressive models are prone to repetitive degeneration, especially with protein sequences. The `batch_inference.py` script includes six mitigations that are active by default:

1. **Repetition penalty** (Keskar et al., 2019): Scales down logits of previously generated tokens
2. **Frequency penalty**: Linearly penalizes tokens proportional to their occurrence count
3. **Top-p / nucleus sampling** (Holtzman et al., 2020): Dynamic vocabulary truncation
4. **N-gram blocking**: Prevents any n-gram from appearing more than once
5. **Max consecutive repeats**: Caps single amino acid runs (e.g., blocks AAAA)
6. **Max motif repeat**: Caps di/tri-peptide motif repetitions (e.g., blocks AEAEAE)

#### Prompt Format

Each line of the input CSV should contain a protein sequence ending with `>` (the FASTA header character used during training). For example:

```
MKAYLVGSGTRGSLPRAIAEQLAQEG...VKQWALKLND>
MRREVLHSPTTDDAYDASAS...CKGSVSRGGSGHPG>
```

The `>` signals to the model that a new sequence follows, prompting it to generate a continuation. Output is trimmed at `>` or `<` characters.

### Training

#### Prepare data

Place your protein sequences in FASTA format in the appropriate data directory, then run the tokenizer:

```bash
python data/char_dark/prepare.py
```

This creates `train.bin`, `val.bin`, and `meta.pkl` (character-to-integer mappings).

#### Train a model

```bash
# Single GPU
python train_.py config/train_par_gpt2-s_scratch.py

# Multi-GPU (2 GPUs)
torchrun --standalone --nproc_per_node=2 train_.py config/train_par_gpt2-s_scratch.py
```

#### Training Configuration

The default configuration in `config/train_par_gpt2-s_scratch.py` trains a GPT-2 small model:

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 (12 layers, 12 heads, 768 embedding) |
| Context length | 1024 characters |
| Batch size | 12 |
| Learning rate | 5e-4 |
| Max iterations | 40,000 |
| Dropout | 0.2 |
| Optimizer | AdamW (beta2=0.99) |

Modify the config file or pass overrides on the command line:

```bash
python train_.py config/train_par_gpt2-s_scratch.py --max_iters=80000 --dropout=0.1
```

## Datasets

The three datasets correspond to proteome partitions from the TARA-Oceans LA4SR pipeline:

| Dataset | Directory | Description |
|---------|-----------|-------------|
| Dark | `data/char_dark/` | Proteins with no Pfam-A hits |
| Annotated | `data/char_white/` | Proteins with Pfam-A annotations |
| Algae | `data/char_algae/` | Combined sampled proteome |

Each dataset directory contains:
- `input.txt`: Concatenated protein sequences in character format
- `prepare.py`: Script to generate `train.bin`, `val.bin`, and `meta.pkl`
- `meta.pkl`: Character vocabulary with `stoi` (string-to-integer) and `itos` (integer-to-string) mappings

## Citation

If you use this software, please cite:

```
Nelson, Plouviez & Salehi-Ashtiani (2026). ELF-NET: Protein language models
and satellite embeddings for marine microalgal functional composition.
```

This software is built on nanoGPT:

```
Karpathy, A. (2022). nanoGPT. https://github.com/karpathy/nanoGPT
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The original nanoGPT framework is by Andrej Karpathy (MIT License, 2022). Modifications and additions for protein language modeling are by Kourosh Salehi-Ashtiani (MIT License, 2026).
