import time

out_dir = 'out-shakespeare'
eval_interval = 50
eval_iters = 10
wandb_log = False # feel free to turn on
wandb_project = 'Classification4'
wandb_run_name = 'algaGPT2-S_set4x2_resume'

dataset = 'shakespeare_char'
init_from = 'resume' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves

always_save_checkpoint = False

# the number of examples per iter:
# 10 batch_size * 1 grad_accum * 1024 tokens = 10240 tokens/iter
# has xx tokens, so 1 epoch ~= xx iters
batch_size = 48
gradient_accumulation_steps = 2
max_iters = 2000000

# finetune at constant LR
learning_rate = 1e-5
decay_lr = False

# To resume training (1GPU) a ckpt in 'shakespear-out' run this 'python train_gpt2-S_resume.py config/finetune_label3.py'
# To resume training GPT2-M or -S on 2 GPUs: torchrun --standalone --nproc_per_node=2 train_.py config/finetune_label3.py
