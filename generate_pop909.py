"""Generate melody from POP909 chord features and write MIDI plus visualization."""

from __future__ import annotations

import argparse
import os
from typing import List

import mido
import numpy as np
import torch

from pop909_utils import HOLD_TOKEN, REST_TOKEN, format_chord_feature, generate_tokens, load_checkpoint


def load_condition(args) -> np.ndarray:
    if args.chord_npy:
        chord = np.load(args.chord_npy)
        if chord.ndim == 3:
            chord = chord[args.test_index]
        return chord.astype(np.int64)

    path = os.path.join(args.data_dir, "test_chord.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run preprocess_pop909.py first or pass --chord_npy.")
    chord = np.load(path)
    if len(chord) == 0:
        raise RuntimeError("test_chord.npy is empty.")
    index = min(max(args.test_index, 0), len(chord) - 1)
    return chord[index].astype(np.int64)


def melody_to_midi(tokens: np.ndarray, midi_path: str, bpm: float = 120.0, velocity: int = 84) -> None:
    ticks_per_beat = 480
    step_ticks = ticks_per_beat // 4
    tempo = mido.bpm2tempo(bpm)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("track_name", name="generated_melody", time=0))

    pending_time = 0
    active_pitch = None
    active_start = 0

    def close_active(current_tick: int) -> None:
        nonlocal pending_time, active_pitch, active_start
        if active_pitch is None:
            return
        duration = max(step_ticks, current_tick - active_start)
        track.append(mido.Message("note_off", note=int(active_pitch), velocity=0, time=duration if pending_time == 0 else pending_time))
        pending_time = 0
        active_pitch = None

    current_tick = 0
    for token in tokens:
        token = int(token)
        if 0 <= token <= 127:
            if active_pitch is not None:
                close_active(current_tick)
            track.append(mido.Message("note_on", note=token, velocity=velocity, time=pending_time))
            pending_time = 0
            active_pitch = token
            active_start = current_tick
        elif token == REST_TOKEN:
            if active_pitch is not None:
                close_active(current_tick)
            pending_time += step_ticks
        elif token == HOLD_TOKEN:
            if active_pitch is None:
                pending_time += step_ticks
        current_tick += step_ticks

    if active_pitch is not None:
        close_active(current_tick)
    track.append(mido.MetaMessage("end_of_track", time=0))
    os.makedirs(os.path.dirname(midi_path), exist_ok=True)
    mid.save(midi_path)


def save_plot(tokens: np.ndarray, chord: np.ndarray, plot_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skipping plot: {exc}")
        return

    steps = np.arange(len(tokens))
    pitches = np.full(len(tokens), np.nan)
    rest = tokens == REST_TOKEN
    hold = tokens == HOLD_TOKEN
    note_mask = (tokens >= 0) & (tokens <= 127)
    pitches[note_mask] = tokens[note_mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(steps[note_mask], pitches[note_mask], s=26, label="onset", color="#1f77b4")
    ax.scatter(steps[hold], np.full(np.sum(hold), 48), s=8, label="hold", color="#7f7f7f", alpha=0.45)
    ax.scatter(steps[rest], np.full(np.sum(rest), 45), s=8, label="rest", color="#d62728", alpha=0.35)
    ax.set_xlabel("16th-note step")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Generated POP909 melody tokens")
    ax.set_ylim(40, 90)
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right")

    chord_labels: List[str] = []
    last = None
    for idx, feat in enumerate(chord):
        label = format_chord_feature(feat)
        if idx % 16 == 0 or label != last:
            chord_labels.append(f"{idx}:{label}")
            last = label
    ax.text(0.01, -0.28, " | ".join(chord_labels[:24]), transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_chord_log(chord: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("step,chord,beat_position,bar_position\n")
        last = None
        for idx, feat in enumerate(chord):
            label = format_chord_feature(feat)
            if label != last or idx % 16 == 0:
                f.write(f"{idx},{label},{int(feat[2])},{int(feat[3])}\n")
                last = label


def generate(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    chord = load_condition(args)
    tokens = generate_tokens(
        model,
        chord,
        device,
        decoding=args.decoding,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{checkpoint['model_name']}_{args.decoding}_idx{args.test_index}"
    midi_path = os.path.join(args.out_dir, stem + ".mid")
    plot_path = os.path.join(args.out_dir, stem + ".png")
    token_path = os.path.join(args.out_dir, stem + "_tokens.npy")
    chord_log_path = os.path.join(args.out_dir, stem + "_chords.csv")

    melody_to_midi(tokens, midi_path, bpm=args.bpm)
    save_plot(tokens, chord, plot_path)
    save_chord_log(chord, chord_log_path)
    np.save(token_path, tokens)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Generated MIDI: {midi_path}")
    print(f"Token array: {token_path}")
    print(f"Chord log: {chord_log_path}")
    print(f"Plot: {plot_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate POP909 melody from chord progression.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="processed_pop909")
    parser.add_argument("--out_dir", default="generated_pop909")
    parser.add_argument("--test_index", type=int, default=0)
    parser.add_argument("--chord_npy", default=None, help="Optional .npy chord feature file.")
    parser.add_argument("--decoding", choices=["greedy", "top_k"], default="greedy")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    generate(build_arg_parser().parse_args())
