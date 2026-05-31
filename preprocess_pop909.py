"""Preprocess POP909 for chord-conditioned symbolic melody generation.

Expected input directory:
  data/POP909/
    001/
      001.mid
      chord_midi.txt
      beat_midi.txt
      key_audio.txt or key_midi.txt   (optional)
    002/
      ...

The script is intentionally tolerant of common POP909 mirrors. It searches for
song folders, chord annotation text files, beat annotation text files, and a MIDI
track named MELODY/lead/vocal. Songs without chord annotation or melody notes are
skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mido
import numpy as np

from pop909_utils import (
    CHORD_FEATURE_DIM,
    HOLD_TOKEN,
    NONE_QUALITY,
    REST_TOKEN,
    UNKNOWN_ROOT,
    format_chord_feature,
    parse_chord_symbol,
    save_vocab,
)


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int


@dataclass
class ChordSegment:
    start: float
    end: float
    root: int
    quality: int
    symbol: str


def warn(message: str, warnings: List[str]) -> None:
    warnings.append(message)
    print(f"[WARN] {message}")


def list_song_dirs(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"POP909 directory not found: {data_dir}")
    dirs = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    return dirs


def find_file(song_dir: str, patterns: Sequence[str], suffixes: Sequence[str]) -> Optional[str]:
    candidates = []
    for root, _dirs, files in os.walk(song_dir):
        for filename in files:
            lower = filename.lower()
            if any(pattern in lower for pattern in patterns) and any(lower.endswith(s) for s in suffixes):
                candidates.append(os.path.join(root, filename))
    return sorted(candidates)[0] if candidates else None


def find_song_midi(song_dir: str) -> Optional[str]:
    basename = os.path.basename(song_dir)
    direct = os.path.join(song_dir, f"{basename}.mid")
    if os.path.exists(direct):
        return direct
    direct = os.path.join(song_dir, f"{basename}.midi")
    if os.path.exists(direct):
        return direct
    midi_files = []
    for root, _dirs, files in os.walk(song_dir):
        for filename in files:
            if filename.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, filename))
    return sorted(midi_files)[0] if midi_files else None


def read_tempo_events(mid: mido.MidiFile) -> List[Tuple[int, int]]:
    events = [(0, 500_000)]
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append((tick, msg.tempo))
    events.sort(key=lambda x: x[0])
    compact = []
    for tick, tempo in events:
        if compact and compact[-1][0] == tick:
            compact[-1] = (tick, tempo)
        else:
            compact.append((tick, tempo))
    return compact


def tick_to_seconds(abs_tick: int, ticks_per_beat: int, tempo_events: Sequence[Tuple[int, int]]) -> float:
    seconds = 0.0
    last_tick = 0
    tempo = tempo_events[0][1] if tempo_events else 500_000
    for tick, new_tempo in tempo_events[1:]:
        if abs_tick <= tick:
            break
        seconds += mido.tick2second(tick - last_tick, ticks_per_beat, tempo)
        last_tick = tick
        tempo = new_tempo
    seconds += mido.tick2second(abs_tick - last_tick, ticks_per_beat, tempo)
    return seconds


def extract_track_notes(mid_path: str, warnings: List[str]) -> List[Note]:
    try:
        mid = mido.MidiFile(mid_path)
    except Exception as exc:
        warn(f"Could not read MIDI {mid_path}: {exc}", warnings)
        return []

    tempo_events = read_tempo_events(mid)
    tracks: List[Tuple[str, List[Note]]] = []

    for track in mid.tracks:
        abs_tick = 0
        track_name = ""
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        notes: List[Note] = []
        for msg in track:
            abs_tick += msg.time
            if msg.type == "track_name":
                track_name = str(msg.name)
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel if hasattr(msg, "channel") else 0, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type in {"note_off", "note_on"}:
                key = (msg.channel if hasattr(msg, "channel") else 0, msg.note)
                if key in active:
                    start_tick, velocity = active.pop(key)
                    start = tick_to_seconds(start_tick, mid.ticks_per_beat, tempo_events)
                    end = tick_to_seconds(abs_tick, mid.ticks_per_beat, tempo_events)
                    if end > start:
                        notes.append(Note(start, end, int(msg.note), int(velocity)))
        tracks.append((track_name, sorted(notes, key=lambda n: (n.start, -n.pitch))))

    named = [
        notes
        for name, notes in tracks
        if notes and any(keyword in name.lower() for keyword in ("melody", "lead", "vocal", "voice"))
    ]
    if named:
        return named[0]

    nonempty = [notes for _name, notes in tracks if notes]
    if nonempty:
        # POP909 MIDI files usually place the melody in the first note-bearing track.
        return nonempty[0]
    return []


def parse_annotation_line(line: str) -> Optional[Tuple[float, float, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = re.split(r"[\s,]+", stripped)
    floats: List[Tuple[int, float]] = []
    for idx, part in enumerate(parts):
        try:
            floats.append((idx, float(part)))
        except ValueError:
            continue
    if len(floats) < 2:
        return None
    start_idx, start = floats[0]
    end_idx, end = floats[1]
    symbol_parts = [p for i, p in enumerate(parts) if i not in {start_idx, end_idx}]
    symbol = symbol_parts[-1] if symbol_parts else "N"
    if end <= start:
        return None
    return start, end, symbol


def read_chords(song_dir: str, warnings: List[str]) -> List[ChordSegment]:
    chord_path = find_file(song_dir, patterns=("chord_midi", "chord"), suffixes=(".txt", ".lab", ".csv"))
    if chord_path is None:
        warn(f"Missing chord annotation in {song_dir}", warnings)
        return []
    segments: List[ChordSegment] = []
    with open(chord_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is None:
                continue
            start, end, symbol = parsed
            root, quality = parse_chord_symbol(symbol)
            segments.append(ChordSegment(start, end, root, quality, symbol))
    if not segments:
        warn(f"No valid chord segments in {chord_path}", warnings)
    return sorted(segments, key=lambda x: x.start)


def read_beat_times(song_dir: str, warnings: List[str]) -> Optional[List[float]]:
    beat_path = find_file(song_dir, patterns=("beat_midi", "beat"), suffixes=(".txt", ".csv"))
    if beat_path is None:
        warn(f"Missing beat annotation in {song_dir}; falling back to 120 BPM grid", warnings)
        return None
    times: List[float] = []
    with open(beat_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", stripped)
            for part in parts:
                try:
                    times.append(float(part))
                    break
                except ValueError:
                    continue
    times = sorted(set(t for t in times if t >= 0.0))
    if len(times) < 2:
        warn(f"Not enough beat times in {beat_path}; falling back to 120 BPM grid", warnings)
        return None
    return times


def read_key_annotation(song_dir: str) -> Optional[Dict[str, str]]:
    key_path = find_file(song_dir, patterns=("key_midi", "key_audio", "key"), suffixes=(".txt", ".csv"))
    if key_path is None:
        return None
    entries = []
    with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed = parse_annotation_line(stripped)
            if parsed is not None:
                start, end, symbol = parsed
                entries.append({"start": start, "end": end, "key": symbol})
            else:
                entries.append({"key": stripped})
    return {"path": key_path, "entries": entries}


def make_grid_times(beat_times: Optional[List[float]], max_time: float, fallback_bpm: float = 120.0) -> np.ndarray:
    if beat_times and len(beat_times) >= 2:
        times: List[float] = []
        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            interval = beat_times[i + 1] - start
            if interval <= 0:
                continue
            for sub in range(4):
                times.append(start + interval * sub / 4.0)
        last_interval = np.median(np.diff(beat_times))
        next_time = beat_times[-1]
        while next_time <= max_time + last_interval:
            for sub in range(4):
                times.append(next_time + last_interval * sub / 4.0)
            next_time += last_interval
        return np.asarray(sorted(t for t in times if t <= max_time + last_interval), dtype=np.float64)

    step_seconds = 60.0 / fallback_bpm / 4.0
    return np.arange(0.0, max_time + step_seconds, step_seconds, dtype=np.float64)


def notes_to_melody_tokens(notes: Sequence[Note], grid_times: np.ndarray) -> np.ndarray:
    melody = np.full(len(grid_times), REST_TOKEN, dtype=np.int64)
    if len(grid_times) < 2:
        return melody
    step = float(np.median(np.diff(grid_times)))

    for note in sorted(notes, key=lambda n: (n.start, -n.pitch)):
        start_idx = int(np.searchsorted(grid_times, note.start, side="right") - 1)
        start_idx = max(0, min(start_idx, len(grid_times) - 1))
        end_idx = int(np.searchsorted(grid_times, note.end, side="left"))
        end_idx = max(start_idx + 1, min(end_idx, len(grid_times)))

        current = melody[start_idx]
        if current in {REST_TOKEN, HOLD_TOKEN} or note.pitch > current:
            melody[start_idx] = int(note.pitch)
        for idx in range(start_idx + 1, end_idx):
            if melody[idx] == REST_TOKEN:
                melody[idx] = HOLD_TOKEN
    return melody


def chords_to_features(chords: Sequence[ChordSegment], grid_times: np.ndarray) -> np.ndarray:
    features = np.zeros((len(grid_times), CHORD_FEATURE_DIM), dtype=np.int64)
    starts = [segment.start for segment in chords]
    for idx, time in enumerate(grid_times):
        root = UNKNOWN_ROOT
        quality = NONE_QUALITY
        chord_idx = bisect_right(starts, float(time)) - 1
        if 0 <= chord_idx < len(chords):
            segment = chords[chord_idx]
            if segment.start <= time < segment.end:
                root = segment.root
                quality = segment.quality
        bar_position = idx % 16
        beat_position = (idx // 4) % 4
        features[idx] = [root, quality, beat_position, bar_position]
    return features


def make_windows(chord: np.ndarray, melody: np.ndarray, seq_len: int, stride: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    chord_windows: List[np.ndarray] = []
    melody_windows: List[np.ndarray] = []
    for start in range(0, max(0, len(melody) - seq_len + 1), stride):
        chord_windows.append(chord[start : start + seq_len])
        melody_windows.append(melody[start : start + seq_len])
    return chord_windows, melody_windows


def split_song_indices(num_songs: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, set]:
    indices = np.arange(num_songs)
    train_end = int(num_songs * train_ratio)
    val_end = train_end + int(num_songs * val_ratio)
    return {
        "train": set(indices[:train_end].tolist()),
        "val": set(indices[train_end:val_end].tolist()),
        "test": set(indices[val_end:].tolist()),
    }


def process_pop909(args) -> None:
    warnings: List[str] = []
    song_dirs = list_song_dirs(args.data_dir)
    if args.max_songs:
        song_dirs = song_dirs[: args.max_songs]
    if not song_dirs:
        raise RuntimeError(f"No song directories found in {args.data_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    splits = split_song_indices(len(song_dirs))
    arrays = {split: {"chord": [], "melody": []} for split in ("train", "val", "test")}
    metadata = []
    processed_songs = 0

    for song_idx, song_dir in enumerate(song_dirs):
        midi_path = find_song_midi(song_dir)
        if midi_path is None:
            warn(f"Missing MIDI file in {song_dir}; skipping song", warnings)
            continue
        chords = read_chords(song_dir, warnings)
        if not chords:
            warn(f"Skipping {song_dir} because chord annotation is unavailable", warnings)
            continue
        notes = extract_track_notes(midi_path, warnings)
        if not notes:
            warn(f"Skipping {song_dir} because no melody/lead notes were found", warnings)
            continue

        beat_times = read_beat_times(song_dir, warnings)
        key_annotation = read_key_annotation(song_dir)
        max_time = max(max(note.end for note in notes), max(chord.end for chord in chords))
        grid_times = make_grid_times(beat_times, max_time=max_time)
        if len(grid_times) < args.seq_len:
            warn(f"Skipping {song_dir}: only {len(grid_times)} grid steps", warnings)
            continue

        melody = notes_to_melody_tokens(notes, grid_times)
        chord = chords_to_features(chords, grid_times)
        chord_windows, melody_windows = make_windows(chord, melody, args.seq_len, args.stride)
        if not melody_windows:
            warn(f"Skipping {song_dir}: no full windows created", warnings)
            continue

        split = "test"
        for name, idx_set in splits.items():
            if song_idx in idx_set:
                split = name
                break
        arrays[split]["chord"].extend(chord_windows)
        arrays[split]["melody"].extend(melody_windows)
        metadata.append(
            {
                "song_id": os.path.basename(song_dir),
                "split": split,
                "midi_path": midi_path,
                "num_windows": len(melody_windows),
                "has_beat_annotation": beat_times is not None,
                "key_annotation": key_annotation,
            }
        )
        processed_songs += 1

        if args.verbose:
            first_chord = format_chord_feature(chord[0])
            print(
                f"{os.path.basename(song_dir)} -> {split}: "
                f"{len(melody_windows)} windows, first chord {first_chord}"
            )

    for split in ("train", "val", "test"):
        chord_arr = np.asarray(arrays[split]["chord"], dtype=np.int64)
        melody_arr = np.asarray(arrays[split]["melody"], dtype=np.int64)
        if chord_arr.size == 0:
            chord_arr = np.zeros((0, args.seq_len, CHORD_FEATURE_DIM), dtype=np.int64)
            melody_arr = np.zeros((0, args.seq_len), dtype=np.int64)
        np.save(os.path.join(args.out_dir, f"{split}_chord.npy"), chord_arr)
        np.save(os.path.join(args.out_dir, f"{split}_melody.npy"), melody_arr)
        print(f"Saved {split}: chord {chord_arr.shape}, melody {melody_arr.shape}")

    save_vocab(os.path.join(args.out_dir, "vocab_pop909.json"), args.seq_len)
    with open(os.path.join(args.out_dir, "preprocess_warnings.log"), "w", encoding="utf-8") as f:
        for message in warnings:
            f.write(message + "\n")
    with open(os.path.join(args.out_dir, "song_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Processed {processed_songs}/{len(song_dirs)} songs. Warnings: {len(warnings)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess POP909 into chord/melody token arrays.")
    parser.add_argument("--data_dir", default="data/POP909", help="Path to raw POP909 directory.")
    parser.add_argument("--out_dir", default="processed_pop909", help="Output directory for numpy arrays.")
    parser.add_argument("--seq_len", type=int, default=64, help="Fixed segment length in 16th-note steps.")
    parser.add_argument("--stride", type=int, default=64, help="Window stride in 16th-note steps.")
    parser.add_argument("--max_songs", type=int, default=None, help="Quick test mode: process only first N songs.")
    parser.add_argument("--verbose", action="store_true", help="Print per-song progress.")
    return parser


if __name__ == "__main__":
    process_pop909(build_arg_parser().parse_args())
