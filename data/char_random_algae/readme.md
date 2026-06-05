# char_random_algae

Random-control dataset: protein sequences whose residues are drawn i.i.d. from
the **empirical amino-acid frequencies of the algal proteome** (composition-
matched), with sequence lengths sampled from the empirical algal length
distribution. Training on this set establishes a **composition-matched floor**
for the LLM training loss: the model can learn the single-residue (unigram)
bias but no higher-order structure, so the floor sits slightly below the
`char_random_uniform` floor.

Pair this with `char_random_uniform` (same lengths, uniform residues) to
separate composition bias from higher-order sequence structure. If the real
dark / annotated / algae loss curves drop below *both* floors, that is evidence
of genuine higher-order structure beyond amino-acid composition.

## Generate

```bash
# from the repo root
python generate_random_sequences.py -n 400000 --seed 1337 -c algae \
    -o data/char_random_algae/input.txt

# prepare.py is a character-level tokenizer
python data/char_random_algae/prepare.py   # -> train.bin, val.bin, meta.pkl
```
