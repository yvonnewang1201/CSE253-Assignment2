# CSE253 Assignment 2 Handoff

Task 1 remains the original MAESTRO unconditioned symbolic generation pipeline in `preprocess.py`.
Task 2 was refactored into **POP909 chord-conditioned symbolic melody generation**.

## Goal

Question: does chord conditioning improve symbolic melody generation compared with an unconditional melody language model?

```text
Input : chord_root + chord_quality + beat_position + bar_position
Output: monophonic melody tokens
Tokens: 0-127 = MIDI pitch onset, 128 = REST, 129 = HOLD
```

POP909 is used because MAESTRO has no explicit melody/chord labels, while POP909 includes MIDI plus chord, beat, and key annotations.

## Data

Expected structure:

```text
data/POP909/
  001/
    001.mid
    chord_midi.txt
    beat_midi.txt
    key_audio.txt
  002/
  ...
```

Current local POP909 copy was downloaded from `music-x-lab/POP909-Dataset` and has 909 song folders.

## Added Files

```text
preprocess_pop909.py   # 16th-note chord/melody arrays
pop909_utils.py        # vocab, models, generation, metrics
train_pop909.py        # trains all models with early stopping
generate_pop909.py     # MIDI + plot + chord log generation
evaluate_pop909.py     # loss, harmony, distribution metrics
```

Generated outputs are git-ignored: `processed_pop909/`, `checkpoints/`, `generated_pop909/`, `evaluation_pop909/`.

## Run From Scratch

```bash
conda activate ece176
python preprocess_pop909.py --data_dir data/POP909 --out_dir processed_pop909
python train_pop909.py --device cuda
python generate_pop909.py --checkpoint checkpoints/chord_gru_best.pt --decoding top_k --top_k 8 --device cuda
python evaluate_pop909.py --unconditional_checkpoint checkpoints/unconditional_gru_best.pt --chord_checkpoint checkpoints/chord_gru_best.pt --transformer_checkpoint checkpoints/chord_transformer_best.pt --decoding top_k --top_k 8 --device cuda
```

`python train_pop909.py` trains all three models by default: `unconditional_gru`, `chord_gru`, `chord_transformer`.

Default training:

```text
epochs = 20, early_stop_patience = 4, batch_size = 128
GRU hidden_dim = 128, layers = 1
Transformer hidden_dim = 128, layers = 2, heads = 4
unconditional loss weights = 1.0/1.0/1.0
conditioned loss weights = onset/rest/hold 2.0/0.9/0.9
```

## Current Full-Data Run

Preprocessing:

```text
Processed 909/909 songs, warnings: 0
train: (17702, 64, 4), melody: (17702, 64)
val:   (2092, 64, 4), melody: (2092, 64)
test:  (2176, 64, 4), melody: (2176, 64)
```

Latest evaluation:

```text
Real: onset 0.2222, rest 0.3318, chord-tone 0.4509
unconditional_gru: loss 0.8862, ppl 2.43, chord-tone 0.2324, rest 0.6348, onset 0.1243, pc-js 0.2621, int-js 0.0199, bar-js 0.0126
chord_gru:         loss 0.8975, ppl 2.45, chord-tone 0.4783, rest 0.2153, onset 0.3379, pc-js 0.0528, int-js 0.0246, bar-js 0.0193
chord_transformer: loss 0.8977, ppl 2.45, chord-tone 0.5226, rest 0.1501, onset 0.4075, pc-js 0.0596, int-js 0.0158, bar-js 0.0258
```

Interpretation: chord-conditioned models improve harmonic consistency and pitch-class similarity. They still generate denser melodies than real POP909, so rhythm/rest realism remains a limitation.

## Generated Samples

```text
generated_pop909/chord_gru_top_k_idx0.mid
generated_pop909/chord_gru_top_k_idx1.mid
generated_pop909/chord_gru_top_k_idx2.mid
generated_pop909/chord_transformer_top_k_idx0.mid
```

Each sample also has a `.png` piano-roll plot, `.npy` token file, and `.csv` chord log in `generated_pop909/`.
