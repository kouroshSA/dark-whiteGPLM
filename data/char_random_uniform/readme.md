# char_random_uniform

Random-control dataset: protein sequences whose residues are drawn i.i.d.
**uniformly** over the 20 standard amino acids, with sequence lengths sampled
from the empirical algal length distribution. Training on this set establishes a
**maximum-entropy floor** for the LLM training loss (cross-entropy converges to
~ln(20) ~= 3.0 nats), against which the dark / annotated / algae loss curves are
interpreted.

Pair this with `char_random_algae` (same lengths, but residue frequencies
matched to the algal proteome) to separate composition bias from higher-order
sequence structure.

## Generate

```bash
# from the repo root
python generate_random_sequences.py -n 400000 --seed 1337 -c uniform \
    -o data/char_random_uniform/input.txt

# prepare.py is a character-level tokenizer
python data/char_random_uniform/prepare.py   # -> train.bin, val.bin, meta.pkl
```
