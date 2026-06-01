out_dir = 'out_dark_40ki_standard'
eval_interval = 1000 # keep frequent because we'll overfit
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
# override via command line if you like
wandb_project = 'dark-light'
wandb_run_name = 'dark_40ki_standard'
dataset = 'char_dark'
init_from = 'scratch' # this is the largest GPT-2 model
gradient_accumulation_steps = 2
batch_size = 12

block_size = 1024 # context of up to n previous characters


# GPT2-M models
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

# using above parameters, gradient = 10, batch = 16, token/iter = 98304, epoch = 97600


learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 40000
lr_decay_iters = 40000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 200 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
