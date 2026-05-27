# CSE 253 Assignment 2 — Symbolic Music Generation

Generate music using deep learning on the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset (~200 hours of professional piano performances).

**Tasks chosen:**
- **Task 1 — Symbolic, unconditioned generation:** Train a model on MIDI event sequences and sample new piano pieces from scratch.
- **Task 2 — Symbolic, conditioned generation:** Generate a full piano arrangement conditioned on a extracted melody line.

---

## Dataset

[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) — 1,276 MIDI files, pre-split by the dataset authors:

| Split | Files | Hours |
|-------|------:|------:|
| train | 962 | ~172 |
| validation | 137 | ~14 |
| test | 177 | ~15 |
| **total** | **1,276** | **~198** |

Download the MIDI-only zip (`maestro-v3.0.0-midi.zip`) from the MAESTRO website and unzip it so the directory structure is:
```
Assignment2/
  maestro-v3.0.0/          ← unzipped MIDI files
    2004/
    2006/
    ...
  maestro-v3.0.0.csv       ← metadata / split labels
```

---

## Setup

### 1. Create a virtual environment (arm64, Python 3.14)

```bash
arch -arm64 python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Jupyter (to run the notebook)

```bash
pip install jupyter
```

---

## Preprocessing

The preprocessing script converts raw MIDI files into token sequences for model training.

```bash
# Full dataset (~30 min)
python preprocess.py

# Quick test — 5 files per split (~30 sec)
python preprocess.py 5
```

Output lands in `processed/`:

| File | Shape | Description |
|------|-------|-------------|
| `{split}_task1.npy` | (N, 512) int32 | Windowed event token sequences for Task 1 |
| `{split}_melody.npy` | (N, 512) int16 | Melody condition tokens for Task 2 |
| `{split}_full.npy` | (N, 512) int32 | Full-piece target tokens for Task 2 |
| `vocab.pkl` / `vocab.txt` | — | Token ↔ name mapping |

### Token vocabulary (size 389)

| Range | Meaning |
|-------|---------|
| 0–127 | `NOTE_ON` pitch |
| 128–255 | `NOTE_OFF` pitch |
| 256–355 | `TIME_SHIFT` (10 ms steps, up to 1 s per token) |
| 356–387 | `VELOCITY` (128 levels binned into 32) |
| 388 | `PAD` |

---

## Project Structure

```
Assignment2/
├── preprocess.py          # MIDI parsing & tokenization
├── requirements.txt       # Python dependencies
├── workbook.ipynb         # Main Jupyter notebook (submitted as workbook.html)
├── processed/             # Preprocessed numpy arrays (git-ignored)
├── maestro-v3.0.0/        # Raw MIDI dataset (git-ignored)
├── maestro-v3.0.0.csv     # Dataset metadata
└── README.md
```

---

## References

- Oore et al. (2018). *This Time with Feeling: Learning Expressive Musical Performance.* — performance event encoding used for tokenization.
- Hawthorne et al. (2019). *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.* ICLR 2019.
