#!/usr/bin/env python3
"""
bro-chat.py - Interactive Chat Interface for broGPT

A nanochat-inspired interactive chat interface that supports both:
- Vanilla GPT models (model.py)
- HOPE architecture models with Titan memory (model_hope_v2.py)

Features:
- Interactive command-line chat
- Streaming token generation
- Conversation history
- Special commands (/clear, /quit, /status, /reset)
- Support for HOPE Titan memory and in-context learning
- Automatic model type detection

Based on:
- Karpathy's nanochat (https://github.com/karpathy/nanochat)
- nanoGPT sampling scripts
- HOPE architecture implementation

Usage Examples:
--------------
# Vanilla GPT model:
python bro-chat.py --model_dir out-vanilla

# HOPE model with default settings:
python bro-chat.py --model_dir out

# HOPE model with Titan memory enabled:
python bro-chat.py --model_dir out --use_titan=1

# HOPE model with full features (Titan + surprise updates):
python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1

# Adjust generation parameters:
python bro-chat.py --model_dir out --temperature=0.8 --top_k=50 --max_tokens=512

# Single prompt mode (non-interactive):
python bro-chat.py --model_dir out -p "Hello, how are you?"

Author: Developed with assistance from Claude
"""

import os
import sys
import argparse
import pickle
import random
from contextlib import nullcontext
from typing import Optional, Tuple, Any

import torch
import torch.nn.functional as F

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive chat interface for broGPT (vanilla & HOPE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with vanilla model:
  python bro-chat.py --model_dir out-vanilla

  # HOPE model with Titan memory:
  python bro-chat.py --model_dir out --use_titan=1

  # Full HOPE features:
  python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1

  # Single prompt (non-interactive):
  python bro-chat.py --model_dir out -p "Tell me a story"
        """
    )

    # Model configuration
    parser.add_argument('--model_dir', type=str, default='out',
                        help='Directory containing ckpt.pt checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Data type for inference')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster inference')

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling threshold')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum tokens to generate per response')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')

    # HOPE-specific options
    parser.add_argument('--use_titan', type=int, default=-1,
                        help='Use Titan memory in forward pass (-1=auto, 0=off, 1=on)')
    parser.add_argument('--enable_surprise', type=int, default=0,
                        help='Enable Titan surprise updates (0=off, 1=on)')
    parser.add_argument('--surprise_in_eval', type=int, default=0,
                        help='Allow surprise updates during eval (0=off, 1=on)')

    # Memory state persistence (HOPE only)
    parser.add_argument('--memory_state_in', type=str, default='',
                        help='Path to load memory state from')
    parser.add_argument('--memory_state_out', type=str, default='',
                        help='Path to save memory state to on exit')

    # Conversation mode
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Single prompt (non-interactive mode)')
    parser.add_argument('--no_stream', action='store_true',
                        help='Disable streaming output')

    # UI customization
    parser.add_argument('--user_prefix', type=str, default='You',
                        help='Prefix for user messages')
    parser.add_argument('--bot_prefix', type=str, default='broGPT',
                        help='Prefix for bot responses')

    return parser.parse_args()


# =============================================================================
# Model Loading
# =============================================================================

def detect_model_type(checkpoint: dict) -> str:
    """Detect if checkpoint is HOPE or vanilla GPT."""
    model_args = checkpoint.get('model_args', {})
    if 'titan_hidden_mult' in model_args or 'cms_level_specs' in model_args:
        return 'hope'
    return 'vanilla'


def load_model(args) -> Tuple[Any, dict, str]:
    """Load model from checkpoint, auto-detecting type."""
    ckpt_path = os.path.join(args.model_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # First, load checkpoint to detect type
    # Add LevelSpec to safe globals for HOPE models
    try:
        from model_hope_v2 import LevelSpec
        torch.serialization.add_safe_globals([LevelSpec])
    except ImportError:
        pass

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model_type = detect_model_type(checkpoint)
    model_args = checkpoint['model_args']

    print(f"Detected model type: {model_type.upper()}")

    if model_type == 'hope':
        from model_hope_v2 import HopeGPTConfig, HopeGPT, LevelSpec

        # Rebuild CMS level specs if needed
        if 'cms_level_specs' not in model_args or model_args['cms_level_specs'] is None:
            model_args['cms_level_specs'] = [
                LevelSpec(name="level_0", update_period=1, warmup_steps=0),
                LevelSpec(name="level_1", update_period=10, warmup_steps=100),
                LevelSpec(name="level_2", update_period=100, warmup_steps=500),
            ]

        config = HopeGPTConfig(**model_args)
        model = HopeGPT(config)
    else:
        from model import GPTConfig, GPT
        config = GPTConfig(**model_args)
        model = GPT(config)

    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    return model, checkpoint, model_type


def configure_hope_model(model, args, model_type: str):
    """Configure HOPE-specific settings."""
    if model_type != 'hope':
        return

    if hasattr(model, 'config'):
        # Override Titan settings based on args
        if args.use_titan >= 0:
            model.config.use_titan_in_forward = bool(args.use_titan)
        if args.enable_surprise:
            model.config.enable_surprise_updates = True
            # Auto-enable surprise_update_in_eval for inference
            if not args.surprise_in_eval:
                args.surprise_in_eval = 1
            model.config.surprise_update_in_eval = bool(args.surprise_in_eval)

        print(f"HOPE config: use_titan_in_forward={model.config.use_titan_in_forward}, "
              f"enable_surprise_updates={model.config.enable_surprise_updates}")


def load_tokenizer(checkpoint: dict, model_dir: str) -> Tuple[callable, callable, dict]:
    """Load character-level or BPE tokenizer."""
    config = checkpoint.get('config', {})
    dataset = config.get('dataset', '')

    # Try character-level tokenizer first
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    if os.path.exists(meta_path):
        print(f"Loading tokenizer from {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        encode = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]
        decode = lambda l: ''.join([itos.get(i, '') for i in l])
        return encode, decode, meta

    # Fall back to tiktoken (GPT-2 BPE)
    try:
        import tiktoken
        print("Using tiktoken GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        return encode, decode, {}
    except ImportError:
        raise RuntimeError("No tokenizer found. Install tiktoken or provide meta.pkl")


# =============================================================================
# Generation
# =============================================================================

def generate_streaming(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_tokens: list = None,
    block_size: int = 1024,
):
    """Generate tokens one at a time, yielding each token."""
    stop_tokens = stop_tokens or []

    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass
        with torch.no_grad():
            logits = model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Yield the token
        token_id = idx_next.item()
        yield token_id

        # Check stop condition
        if token_id in stop_tokens:
            break

        # Append and continue
        idx = torch.cat((idx, idx_next), dim=1)


# =============================================================================
# Chat Interface
# =============================================================================

ENCOURAGEMENTS = [
    "Go ahead, type something!",
    "I'm listening...",
    "What's on your mind?",
    "Tell me more!",
    "Don't be shy, ask me anything!",
    "Ready when you are!",
    "Type your message...",
]

HELP_TEXT = """
Commands:
  /clear    - Clear conversation history
  /reset    - Reset Titan memory (HOPE models only)
  /status   - Show model status
  /help     - Show this help
  /quit     - Exit chat

Tips:
  - Press Ctrl+C to interrupt generation
  - Press Ctrl+D or type /quit to exit
"""


def print_banner(model_type: str, args):
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("         broGPT Interactive Chat")
    print("=" * 60)
    print(f"  Model: {args.model_dir} ({model_type.upper()})")
    print(f"  Device: {args.device} | Dtype: {args.dtype}")
    print(f"  Temperature: {args.temperature} | Top-k: {args.top_k}")
    if model_type == 'hope':
        titan_status = "ON" if getattr(args, '_titan_enabled', False) else "OFF"
        print(f"  Titan Memory: {titan_status}")
    print("-" * 60)
    print("  Type /help for commands, /quit to exit")
    print("=" * 60 + "\n")


def run_chat(model, encode, decode, args, model_type: str, block_size: int):
    """Main chat loop."""
    # Determine device and context
    device = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    if device_type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
        device_type = 'cpu'

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Track Titan status for banner
    args._titan_enabled = (model_type == 'hope' and
                          hasattr(model, 'config') and
                          getattr(model.config, 'use_titan_in_forward', False))

    # Single prompt mode
    if args.prompt:
        with ctx:
            return run_single_prompt(model, encode, decode, args, device, block_size)

    # Interactive mode
    print_banner(model_type, args)

    conversation_history = ""

    try:
        while True:
            # Get user input
            try:
                user_input = input(f"{args.user_prefix} >>> ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            # Handle empty input
            if not user_input:
                print(random.choice(ENCOURAGEMENTS))
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()[0]

                if cmd in ('/quit', '/exit', '/q'):
                    print("Goodbye!")
                    break

                elif cmd == '/clear':
                    conversation_history = ""
                    print("[Conversation cleared]")
                    continue

                elif cmd == '/reset':
                    if model_type == 'hope' and hasattr(model, 'reset_runtime_memory'):
                        model.reset_runtime_memory()
                        print("[Titan memory reset]")
                    else:
                        print("[Reset not available for this model]")
                    continue

                elif cmd == '/status':
                    print(f"\n[Status]")
                    print(f"  Model type: {model_type.upper()}")
                    print(f"  Block size: {block_size}")
                    print(f"  History length: {len(encode(conversation_history))} tokens")
                    if model_type == 'hope' and hasattr(model, 'config'):
                        print(f"  Titan enabled: {model.config.use_titan_in_forward}")
                        print(f"  Surprise updates: {model.config.enable_surprise_updates}")
                    print()
                    continue

                elif cmd == '/help':
                    print(HELP_TEXT)
                    continue

                else:
                    print(f"Unknown command: {cmd}. Type /help for available commands.")
                    continue

            # Build prompt with history
            full_prompt = conversation_history + user_input

            # Encode and handle context overflow
            input_ids = encode(full_prompt)
            if len(input_ids) > block_size - args.max_tokens:
                # Trim from beginning to leave room for generation
                input_ids = input_ids[-(block_size - args.max_tokens):]

            x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

            # Generate response
            print(f"\n{args.bot_prefix} >>> ", end='', flush=True)

            generated_tokens = []
            generated_text = ""  # Track full text for stop sequence detection
            stop_sequences = ['<|endoftext|>', '<|end|>', '\x00']
            stopped = False

            try:
                with ctx:
                    for token_id in generate_streaming(
                        model,
                        x,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k if args.top_k > 0 else None,
                        top_p=args.top_p,
                        block_size=block_size,
                    ):
                        generated_tokens.append(token_id)
                        token_str = decode([token_id])
                        generated_text += token_str

                        # Check for end-of-text markers in accumulated text
                        for stop_seq in stop_sequences:
                            if stop_seq in generated_text:
                                # Truncate at stop sequence
                                generated_text = generated_text.split(stop_seq)[0]
                                stopped = True
                                break

                        if stopped:
                            break

                        if not args.no_stream:
                            print(token_str, end='', flush=True)

            except KeyboardInterrupt:
                print("\n[Generation interrupted]")

            # Use the tracked generated_text which is already truncated at stop sequence
            response = generated_text

            if args.no_stream:
                print(response)

            print("\n" + "-" * 40)

            # Update conversation history
            conversation_history = full_prompt + response

    except KeyboardInterrupt:
        print("\n\nGoodbye!")

    # Save memory state if requested
    if args.memory_state_out and model_type == 'hope' and hasattr(model, 'save_memory_state'):
        model.save_memory_state(args.memory_state_out)
        print(f"[Saved memory state to {args.memory_state_out}]")


def run_single_prompt(model, encode, decode, args, device, block_size):
    """Run single prompt and exit."""
    input_ids = encode(args.prompt)
    if len(input_ids) > block_size:
        input_ids = input_ids[-block_size:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = []
    generated_text = ""  # Track full text for stop sequence detection
    stop_sequences = ['<|endoftext|>', '<|end|>', '\x00']
    stopped = False

    for token_id in generate_streaming(
        model,
        x,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
        block_size=block_size,
    ):
        generated_tokens.append(token_id)
        token_str = decode([token_id])
        generated_text += token_str

        # Check for end-of-text markers in accumulated text
        for stop_seq in stop_sequences:
            if stop_seq in generated_text:
                # Truncate at stop sequence
                generated_text = generated_text.split(stop_seq)[0]
                stopped = True
                break

        if stopped:
            break

        if not args.no_stream:
            print(token_str, end='', flush=True)

    if args.no_stream:
        print(generated_text)
    else:
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), 'big')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.dtype = 'float32'

    # Load model
    model, checkpoint, model_type = load_model(args)

    # Configure HOPE-specific settings
    configure_hope_model(model, args, model_type)

    # Load tokenizer
    encode, decode, meta = load_tokenizer(checkpoint, args.model_dir)

    # Get block size
    model_args = checkpoint.get('model_args', {})
    block_size = model_args.get('block_size', 1024)

    # Load memory state if specified (HOPE only)
    if args.memory_state_in and model_type == 'hope':
        if hasattr(model, 'load_memory_state'):
            model.load_memory_state(args.memory_state_in)
            print(f"Loaded memory state from {args.memory_state_in}")

    # Run chat
    run_chat(model, encode, decode, args, model_type, block_size)


if __name__ == '__main__':
    main()
