"""
MAESTRO Dataset Preprocessing for CSE253 Assignment 2
Symbolic Music Generation (Task 1: unconditioned, Task 2: conditioned)

Event encoding follows Oore et al. (2018) "This Time with Feeling":
  NOTE_ON  0..127   -> tokens   0-127
  NOTE_OFF 0..127   -> tokens 128-255
  TIME_SHIFT 1..100 -> tokens 256-355  (each unit = 10 ms, max 1 s per token)
  VELOCITY 0..31    -> tokens 356-387  (128 values binned into 32 levels)

Vocab size: 388  (+ 1 PAD token at index 388)
"""

import os
import csv
import pickle
import collections
import numpy as np
import mido

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR   = os.path.join(BASE_DIR, "maestro-v3.0.0")
CSV_PATH   = os.path.join(BASE_DIR, "maestro-v3.0.0.csv")
OUT_DIR    = os.path.join(BASE_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tokenization constants ─────────────────────────────────────────────────────
NOTE_ON_OFFSET    = 0          # 0-127
NOTE_OFF_OFFSET   = 128        # 128-255
TIME_SHIFT_OFFSET = 256        # 256-355  (1..100 steps of 10 ms)
VELOCITY_OFFSET   = 356        # 356-387  (32 velocity bins)
PAD_TOKEN         = 388
VOCAB_SIZE        = 389        # 388 event tokens + 1 PAD

N_TIME_STEPS      = 100        # max time-shift tokens (= 1 000 ms)
TIME_STEP_MS      = 10         # ms per time-shift token
N_VELOCITY_BINS   = 32
VELOCITY_BIN_SIZE = 128 // N_VELOCITY_BINS  # = 4

# Training sequence parameters
SEQ_LEN           = 512        # tokens per training window
STRIDE            = 256        # hop between windows (50% overlap)


# ── Vocabulary helpers ─────────────────────────────────────────────────────────

def velocity_to_bin(velocity: int) -> int:
    return min(velocity // VELOCITY_BIN_SIZE, N_VELOCITY_BINS - 1)


def bin_to_velocity(bin_idx: int) -> int:
    return bin_idx * VELOCITY_BIN_SIZE + VELOCITY_BIN_SIZE // 2


def token_to_str(token: int) -> str:
    if token < NOTE_ON_OFFSET + 128:
        return f"NOTE_ON_{token - NOTE_ON_OFFSET}"
    elif token < NOTE_OFF_OFFSET + 128:
        return f"NOTE_OFF_{token - NOTE_OFF_OFFSET}"
    elif token < TIME_SHIFT_OFFSET + N_TIME_STEPS:
        ms = (token - TIME_SHIFT_OFFSET + 1) * TIME_STEP_MS
        return f"TIME_SHIFT_{ms}ms"
    elif token < VELOCITY_OFFSET + N_VELOCITY_BINS:
        return f"VELOCITY_{token - VELOCITY_OFFSET}"
    elif token == PAD_TOKEN:
        return "PAD"
    return f"UNKNOWN_{token}"


# ── MIDI → note list ──────────────────────────────────────────────────────────

def midi_to_notes(midi_path: str):
    """
    Parse a MIDI file into a sorted list of (start_sec, end_sec, pitch, velocity).
    Handles type-0 and type-1 files. Returns empty list on error.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"  [WARN] Could not read {midi_path}: {e}")
        return []

    tempo = 500_000  # default: 120 BPM
    ticks_per_beat = mid.ticks_per_beat

    # Merge all tracks into a single absolute-time event list
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
    events.sort(key=lambda x: x[0])

    notes = []
    active = {}  # pitch -> (start_sec, velocity)
    current_sec = 0.0
    prev_tick = 0

    for abs_tick, msg in events:
        delta_ticks = abs_tick - prev_tick
        current_sec += mido.tick2second(delta_ticks, ticks_per_beat, tempo)
        prev_tick = abs_tick

        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (current_sec, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                start, vel = active.pop(msg.note)
                notes.append((start, current_sec, msg.note, vel))

    # Close any still-active notes at end of file
    for pitch, (start, vel) in active.items():
        notes.append((start, current_sec, pitch, vel))

    notes.sort(key=lambda x: x[0])
    return notes


# ── Note list → event token sequence ─────────────────────────────────────────

def notes_to_tokens(notes):
    """
    Convert sorted (start, end, pitch, velocity) notes into a flat list of
    integer event tokens using the performance encoding.
    """
    if not notes:
        return []

    tokens = []

    # Build a merged timeline of (time_ms, event_type, value)
    # event_type: 'on' or 'off'
    timeline = []
    for start, end, pitch, vel in notes:
        timeline.append((int(round(start * 1000)), "on",  pitch, vel))
        timeline.append((int(round(end   * 1000)), "off", pitch, 0))
    timeline.sort(key=lambda x: (x[0], x[1] == "on"))  # offs before ons at same time

    prev_ms = 0
    current_velocity_bin = -1

    for event_ms, etype, pitch, vel in timeline:
        # Emit TIME_SHIFT tokens to advance time
        delta_ms = event_ms - prev_ms
        while delta_ms > 0:
            shift = min(delta_ms, N_TIME_STEPS * TIME_STEP_MS)
            n_tokens = max(1, round(shift / TIME_STEP_MS))
            n_tokens = min(n_tokens, N_TIME_STEPS)
            tokens.append(TIME_SHIFT_OFFSET + n_tokens - 1)
            delta_ms -= n_tokens * TIME_STEP_MS
        prev_ms = event_ms

        if etype == "on":
            vbin = velocity_to_bin(vel)
            if vbin != current_velocity_bin:
                tokens.append(VELOCITY_OFFSET + vbin)
                current_velocity_bin = vbin
            tokens.append(NOTE_ON_OFFSET + pitch)
        else:
            tokens.append(NOTE_OFF_OFFSET + pitch)

    return tokens


# ── Token sequence → windowed numpy arrays ────────────────────────────────────

def make_windows(token_seq, seq_len=SEQ_LEN, stride=STRIDE):
    """Slide a window over a flat token list, pad the last window if needed."""
    windows = []
    n = len(token_seq)
    for start in range(0, n, stride):
        chunk = token_seq[start : start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [PAD_TOKEN] * (seq_len - len(chunk))
        windows.append(chunk)
        if start + seq_len >= n:
            break
    return windows


# ── Melody extraction (for Task 2 conditioning) ───────────────────────────────

def extract_melody(notes, quantize_ms=125):
    """
    Extract melody as the highest-pitched note active at each quantized time step.
    Returns a list of (time_step_index, pitch) pairs (REST=-1 for silence).
    quantize_ms: grid resolution in ms (125 ms ≈ 16th note at 120 BPM)
    """
    if not notes:
        return []

    max_time_ms = int(max(end for _, end, _, _ in notes) * 1000) + quantize_ms
    n_steps = max_time_ms // quantize_ms + 1
    melody = [-1] * n_steps  # -1 = REST

    for start, end, pitch, _vel in notes:
        s = int(round(start * 1000)) // quantize_ms
        e = int(round(end   * 1000)) // quantize_ms
        for t in range(s, min(e + 1, n_steps)):
            if melody[t] < pitch:
                melody[t] = pitch

    return melody  # list of pitch values (-1 = rest), length = n_steps


def melody_to_tokens(melody):
    """Encode melody as token indices: pitches 0-127 as-is, REST as 128."""
    return [p if p >= 0 else 128 for p in melody]


# ── Dataset statistics (EDA) ──────────────────────────────────────────────────

def compute_dataset_stats(rows):
    """Print exploratory statistics about the MAESTRO dataset."""
    print("=" * 60)
    print("MAESTRO v3.0.0 — Dataset Statistics")
    print("=" * 60)

    splits = collections.Counter(r["split"] for r in rows)
    print(f"\nTotal files : {len(rows)}")
    for split in ("train", "validation", "test"):
        print(f"  {split:12s}: {splits[split]} files")

    durations = [float(r["duration"]) for r in rows]
    print(f"\nDuration (seconds):")
    print(f"  min  = {min(durations):.1f}")
    print(f"  max  = {max(durations):.1f}")
    print(f"  mean = {np.mean(durations):.1f}")
    print(f"  total= {sum(durations)/3600:.1f} hours")

    composers = collections.Counter(r["canonical_composer"] for r in rows)
    print(f"\nTop 10 composers:")
    for name, count in composers.most_common(10):
        print(f"  {count:4d}  {name}")

    years = collections.Counter(r["year"] for r in rows)
    print(f"\nRecording years: {sorted(years.keys())}")
    print("=" * 60)


# ── Main preprocessing pipeline ───────────────────────────────────────────────

def load_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_split(rows, split_name, midi_dir, seq_len=SEQ_LEN, stride=STRIDE,
                  max_files=None, verbose=True):
    """
    Process all MIDI files for a given split.
    Returns:
      windows_task1  : list of token-ID lists, each length seq_len  (Task 1)
      melody_windows : list of melody token lists (Task 2 conditioning)
      full_windows   : list of full token lists   (Task 2 target)
      token_counts   : per-file token count for statistics
    """
    split_rows = [r for r in rows if r["split"] == split_name]
    if max_files:
        split_rows = split_rows[:max_files]

    windows_task1  = []
    melody_windows = []
    full_windows   = []
    token_counts   = []
    pitch_histogram = np.zeros(128, dtype=np.int64)

    print(f"\nProcessing '{split_name}' split ({len(split_rows)} files)…")
    for i, row in enumerate(split_rows):
        midi_path = os.path.join(midi_dir, row["midi_filename"])
        if not os.path.exists(midi_path):
            print(f"  [WARN] Missing: {midi_path}")
            continue

        notes = midi_to_notes(midi_path)
        if not notes:
            continue

        # ── Task 1: full event token stream ──────────────────────────────────
        tokens = notes_to_tokens(notes)
        token_counts.append(len(tokens))
        wins = make_windows(tokens, seq_len=seq_len, stride=stride)
        windows_task1.extend(wins)

        # ── Task 2: melody (condition) + full-piece (target) ─────────────────
        melody   = extract_melody(notes, quantize_ms=125)
        mel_toks = melody_to_tokens(melody)
        # Window melody to same length as full tokens (trim/pad to seq_len)
        for start in range(0, len(tokens), stride):
            full_chunk = tokens[start : start + seq_len]
            if len(full_chunk) < seq_len:
                full_chunk = full_chunk + [PAD_TOKEN] * (seq_len - len(full_chunk))
            # Align melody window to same time range (approximate by proportion)
            m_start = int(start / len(tokens) * len(mel_toks)) if tokens else 0
            mel_chunk = mel_toks[m_start : m_start + seq_len]
            if len(mel_chunk) < seq_len:
                mel_chunk = mel_chunk + [128] * (seq_len - len(mel_chunk))  # 128=REST
            melody_windows.append(mel_chunk)
            full_windows.append(full_chunk)
            if start + seq_len >= len(tokens):
                break

        # Pitch histogram
        for _, _, pitch, _ in notes:
            pitch_histogram[pitch] += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(split_rows)} files done, "
                  f"{len(windows_task1)} windows so far")

    print(f"  Done. {len(windows_task1)} windows, "
          f"avg tokens/file: {np.mean(token_counts):.0f}" if token_counts else "  Done. 0 windows.")
    return windows_task1, melody_windows, full_windows, token_counts, pitch_histogram


def save_split(out_dir, split_name, windows_task1, melody_windows, full_windows,
               token_counts, pitch_histogram):
    prefix = os.path.join(out_dir, split_name)

    np.save(f"{prefix}_task1.npy",   np.array(windows_task1,  dtype=np.int32))
    np.save(f"{prefix}_melody.npy",  np.array(melody_windows, dtype=np.int16))
    np.save(f"{prefix}_full.npy",    np.array(full_windows,   dtype=np.int32))
    np.save(f"{prefix}_pitch_hist.npy", pitch_histogram)

    with open(f"{prefix}_token_counts.pkl", "wb") as f:
        pickle.dump(token_counts, f)

    print(f"  Saved {split_name} → {out_dir}/")
    print(f"    task1  : {np.array(windows_task1).shape}")
    print(f"    melody : {np.array(melody_windows).shape}")
    print(f"    full   : {np.array(full_windows).shape}")


def save_vocab(out_dir):
    vocab = {token_to_str(i): i for i in range(VOCAB_SIZE)}
    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    # Human-readable version
    with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
        for name, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")
    print(f"\nVocabulary ({VOCAB_SIZE} tokens) saved to {out_dir}/vocab.*")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(max_files_per_split=None):
    rows = load_csv(CSV_PATH)

    # ── EDA ──────────────────────────────────────────────────────────────────
    compute_dataset_stats(rows)

    # ── Save vocabulary ───────────────────────────────────────────────────────
    save_vocab(OUT_DIR)

    # ── Process each split ────────────────────────────────────────────────────
    all_pitch_hist = np.zeros(128, dtype=np.int64)
    for split in ("train", "validation", "test"):
        w1, wm, wf, tc, ph = process_split(
            rows, split, MIDI_DIR,
            seq_len=SEQ_LEN, stride=STRIDE,
            max_files=max_files_per_split,
        )
        save_split(OUT_DIR, split, w1, wm, wf, tc, ph)
        all_pitch_hist += ph

    np.save(os.path.join(OUT_DIR, "pitch_histogram_all.npy"), all_pitch_hist)
    print("\nAll done! Processed data saved to:", OUT_DIR)
    print(f"  SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE}, VOCAB_SIZE={VOCAB_SIZE}")


if __name__ == "__main__":
    import sys
    # Pass an integer as first argument to limit files per split (for quick testing)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_files_per_split=limit)
