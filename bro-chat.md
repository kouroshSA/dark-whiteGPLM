# bro-chat.py - Interactive Chat Interface

  Created at: /home/ksa/Models/ppiGPLM_hope-titan_fixed/bro-chat.py

  Features

  - Interactive command-line chat with streaming token generation
  - Automatic detection of model type (vanilla GPT vs HOPE)
  - Support for HOPE Titan memory and in-context learning
  - Conversation history with context management
  - Special commands: /clear, /reset, /status, /help, /quit
  - Top-k and top-p sampling support
  - Single prompt mode for scripting

  Usage Examples

# Activate conda environment first
  conda activate hope

# Basic interactive chat with HOPE model:
  python bro-chat.py --model_dir out

# HOPE model with Titan memory explicitly enabled:
  python bro-chat.py --model_dir out --use_titan=1

# Full HOPE features (Titan + surprise updates):
  python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1

# Vanilla GPT model (auto-detected):
  python bro-chat.py --model_dir out-vanilla

# Adjust generation parameters:
  python bro-chat.py --model_dir out --temperature=0.8 --top_k=50 --max_tokens=512

# Single prompt mode (non-interactive):
  python bro-chat.py --model_dir out -p "Hello, how are you?"

# With memory state persistence (HOPE only):
  python bro-chat.py --model_dir out --memory_state_out=memory.pt

# Load previous memory state and continue:
  python bro-chat.py --model_dir out --memory_state_in=memory.pt

  Interactive Commands

  - /clear - Clear conversation history
  - /reset - Reset Titan memory (HOPE models only)
  - /status - Show model status
  - /help - Show help
  - /quit - Exit chat



# In-context learning
 Based on the HOPE architecture and the bro-chat.py implementation, here's how to do in-context learning:

  In-Context Learning with HOPE/Titan Memory

  Quick Start

  conda activate hope

  # Enable Titan memory + surprise updates for in-context learning
  python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1

  How It Works

  The HOPE architecture has Titan Memory - an associative memory that learns from context during inference:

  1. Titan Memory (use_titan=1): Adds a learnable memory component to the forward pass
  2. Surprise Updates (enable_surprise=1): When the model is "surprised" (prediction error exceeds threshold), it updates its memory weights in real-time

  Usage Modes

  # 1. Basic in-context learning (memory active in forward pass)
  python bro-chat.py --model_dir out --use_titan=1

  # 2. Full in-context learning (memory updates during conversation)
  python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1

  # 3. Save memory state after learning from context
  python bro-chat.py --model_dir out --use_titan=1 --enable_surprise=1 \
      --memory_state_out=learned_memory.pt

  # 4. Load previously learned memory state
  python bro-chat.py --model_dir out --use_titan=1 \
      --memory_state_in=learned_memory.pt

  # 5. Reset memory during conversation (use /reset command)

  In-Chat Commands

  While chatting:
  - /reset - Reset Titan memory (start fresh)
  - /status - Check if Titan is enabled

  For Batch Processing (Original Script)

  The original sample_fasta3.3_softmax_error_handling3e_hope_v2.py has more advanced options:

  # Prefix adaptation (self-teaching on input before prediction)
  python sample_fasta3.3_softmax_error_handling3e_hope_v2.py \
      --input_file prompts.txt \
      --output_dir results \
      --enable_surprise_updates=1 \
      --adapt_mode=prefix \
      --adapt_steps=5 \
      --reset_policy=per_prompt

  # Save/load memory across sessions
  python sample_fasta3.3_softmax_error_handling3e_hope_v2.py \
      --input_file prompts.txt \
      --memory_state_out=adapted_memory.pt \
      --enable_surprise_updates=1

  Key Parameters

  | Parameter           | Effect                                |
  |---------------------|---------------------------------------|
  | --use_titan=1       | Enable Titan memory in forward pass   |
  | --enable_surprise=1 | Allow memory to update when surprised |
  | --memory_state_in   | Load pre-learned memory               |
  | --memory_state_out  | Save learned memory after session     |

  The model learns patterns from your conversation context and adapts its responses accordingly!

# Note:
  - --use_titan=0 → False (off)
  - --use_titan=1 → True (on)
  - --use_titan=-1 → Auto (uses whatever the checkpoint was trained with)
  - --enable_surprise=0 → False (off)
  - --enable_surprise=1 → True (on)

