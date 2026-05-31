"""Shared utilities for POP909 chord-conditioned melody generation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


MELODY_VOCAB_SIZE = 130
REST_TOKEN = 128
HOLD_TOKEN = 129

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "UNK"]
ROOT_TO_ID = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}
UNKNOWN_ROOT = 12

QUALITY_NAMES = ["major", "minor", "dim", "aug", "sus", "other", "none"]
QUALITY_TO_ID = {name: idx for idx, name in enumerate(QUALITY_NAMES)}
NONE_QUALITY = QUALITY_TO_ID["none"]

CHORD_FEATURE_DIM = 4


def parse_chord_symbol(symbol: str) -> Tuple[int, int]:
    """Map a chord symbol such as C:maj7 or F#:min to root and quality IDs."""
    if not symbol:
        return UNKNOWN_ROOT, NONE_QUALITY

    cleaned = symbol.strip()
    if cleaned in {"N", "NC", "None", "none", "X", "nan"}:
        return UNKNOWN_ROOT, NONE_QUALITY

    root_part = cleaned.split(":", 1)[0].split("/", 1)[0].strip()
    if len(root_part) >= 2 and root_part[1] in {"#", "b"}:
        root = root_part[:2].upper()
    else:
        root = root_part[:1].upper()
    root_id = ROOT_TO_ID.get(root, UNKNOWN_ROOT)

    lower = cleaned.lower()
    if "dim" in lower or ":o" in lower:
        quality = "dim"
    elif "aug" in lower or "+" in lower:
        quality = "aug"
    elif "sus" in lower:
        quality = "sus"
    elif "min" in lower or ":m" in lower or lower.endswith("m"):
        quality = "minor"
    elif (
        "maj" in lower
        or ":m" not in lower
        and root_id != UNKNOWN_ROOT
        and cleaned not in {"N", "NC"}
    ):
        quality = "major"
    else:
        quality = "other"

    return root_id, QUALITY_TO_ID.get(quality, QUALITY_TO_ID["other"])


def chord_tones(root_id: int, quality_id: int) -> Optional[set]:
    """Return pitch classes that belong to a triad-like chord quality."""
    if root_id < 0 or root_id >= 12 or quality_id == NONE_QUALITY:
        return None
    if quality_id == QUALITY_TO_ID["major"]:
        intervals = (0, 4, 7)
    elif quality_id == QUALITY_TO_ID["minor"]:
        intervals = (0, 3, 7)
    elif quality_id == QUALITY_TO_ID["dim"]:
        intervals = (0, 3, 6)
    elif quality_id == QUALITY_TO_ID["aug"]:
        intervals = (0, 4, 8)
    elif quality_id == QUALITY_TO_ID["sus"]:
        intervals = (0, 5, 7)
    else:
        intervals = (0, 4, 7)
    return {(root_id + interval) % 12 for interval in intervals}


def load_vocab(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vocab(path: str, seq_len: int) -> None:
    vocab = {
        "melody": {
            "pitch_onset": "0-127",
            "REST": REST_TOKEN,
            "HOLD": HOLD_TOKEN,
            "vocab_size": MELODY_VOCAB_SIZE,
        },
        "chord_root": {name: idx for idx, name in enumerate(ROOT_NAMES)},
        "chord_quality": {name: idx for idx, name in enumerate(QUALITY_NAMES)},
        "beat_position": {"range": "0-3", "description": "beat index within a 4/4 bar"},
        "bar_position": {"range": "0-15", "description": "16th-note index within a bar"},
        "sequence_length": seq_len,
        "chord_feature_columns": ["root", "quality", "beat_position", "bar_position"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


class MelodyDataset(torch.utils.data.Dataset):
    """Teacher-forced melody dataset with optional chord features."""

    def __init__(self, chord: np.ndarray, melody: np.ndarray):
        if chord.ndim != 3 or chord.shape[-1] != CHORD_FEATURE_DIM:
            raise ValueError(f"Expected chord shape (N, T, 4), got {chord.shape}")
        if melody.ndim != 2:
            raise ValueError(f"Expected melody shape (N, T), got {melody.shape}")
        if chord.shape[:2] != melody.shape:
            raise ValueError(f"Chord and melody shapes do not align: {chord.shape}, {melody.shape}")
        self.chord = chord.astype(np.int64)
        self.melody = melody.astype(np.int64)

    def __len__(self) -> int:
        return self.melody.shape[0]

    def __getitem__(self, idx: int):
        target = self.melody[idx]
        previous = np.empty_like(target)
        previous[0] = REST_TOKEN
        previous[1:] = target[:-1]
        return (
            torch.from_numpy(previous),
            torch.from_numpy(self.chord[idx]),
            torch.from_numpy(target),
        )


class UnconditionalGRU(nn.Module):
    def __init__(self, melody_vocab_size: int = MELODY_VOCAB_SIZE, emb_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(melody_vocab_size, emb_dim)
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, melody_vocab_size)

    def forward(self, previous_melody: torch.Tensor, chord_features: Optional[torch.Tensor] = None):
        x = self.embedding(previous_melody)
        y, _ = self.gru(x)
        return self.output(y)


class ChordGRU(nn.Module):
    def __init__(
        self,
        melody_vocab_size: int = MELODY_VOCAB_SIZE,
        melody_emb_dim: int = 96,
        chord_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.melody_embedding = nn.Embedding(melody_vocab_size, melody_emb_dim)
        self.root_embedding = nn.Embedding(len(ROOT_NAMES), chord_emb_dim)
        self.quality_embedding = nn.Embedding(len(QUALITY_NAMES), chord_emb_dim)
        self.beat_embedding = nn.Embedding(4, 16)
        self.bar_embedding = nn.Embedding(16, 16)
        input_dim = melody_emb_dim + chord_emb_dim * 2 + 32
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, melody_vocab_size)

    def forward(self, previous_melody: torch.Tensor, chord_features: torch.Tensor):
        root = chord_features[..., 0].clamp(0, len(ROOT_NAMES) - 1)
        quality = chord_features[..., 1].clamp(0, len(QUALITY_NAMES) - 1)
        beat = chord_features[..., 2].clamp(0, 3)
        bar = chord_features[..., 3].clamp(0, 15)
        x = torch.cat(
            [
                self.melody_embedding(previous_melody),
                self.root_embedding(root),
                self.quality_embedding(quality),
                self.beat_embedding(beat),
                self.bar_embedding(bar),
            ],
            dim=-1,
        )
        y, _ = self.gru(x)
        return self.output(y)


class ChordTransformer(nn.Module):
    def __init__(
        self,
        melody_vocab_size: int = MELODY_VOCAB_SIZE,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.max_len = max_len
        self.melody_embedding = nn.Embedding(melody_vocab_size, hidden_dim)
        self.root_embedding = nn.Embedding(len(ROOT_NAMES), hidden_dim)
        self.quality_embedding = nn.Embedding(len(QUALITY_NAMES), hidden_dim)
        self.beat_embedding = nn.Embedding(4, hidden_dim)
        self.bar_embedding = nn.Embedding(16, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, melody_vocab_size)

    def forward(self, previous_melody: torch.Tensor, chord_features: torch.Tensor):
        batch, seq_len = previous_melody.shape
        positions = torch.arange(seq_len, device=previous_melody.device).unsqueeze(0).expand(batch, seq_len)
        root = chord_features[..., 0].clamp(0, len(ROOT_NAMES) - 1)
        quality = chord_features[..., 1].clamp(0, len(QUALITY_NAMES) - 1)
        beat = chord_features[..., 2].clamp(0, 3)
        bar = chord_features[..., 3].clamp(0, 15)
        x = (
            self.melody_embedding(previous_melody)
            + self.root_embedding(root)
            + self.quality_embedding(quality)
            + self.beat_embedding(beat)
            + self.bar_embedding(bar)
            + self.position_embedding(positions.clamp(max=self.max_len - 1))
        )
        mask = torch.triu(torch.ones(seq_len, seq_len, device=previous_melody.device), diagonal=1).bool()
        return self.output(self.encoder(x, mask=mask))


def build_model(model_name: str, config: Optional[Dict] = None) -> nn.Module:
    config = dict(config or {})
    if model_name == "unconditional_gru":
        return UnconditionalGRU(**config)
    if model_name == "chord_gru":
        return ChordGRU(**config)
    if model_name == "chord_transformer":
        return ChordTransformer(**config)
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def sequence_loss(model: nn.Module, loader, device: torch.device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for previous, chord, target in loader:
        previous = previous.to(device)
        chord = chord.to(device)
        target = target.to(device)
        logits = model(previous, chord)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    chord_features: np.ndarray,
    device: torch.device,
    decoding: str = "greedy",
    top_k: int = 8,
    temperature: float = 1.0,
) -> np.ndarray:
    """Autoregressively generate one melody sequence for fixed chord features."""
    model.eval()
    chord = torch.as_tensor(chord_features[None, :, :], dtype=torch.long, device=device)
    generated: List[int] = []
    previous = torch.full((1, 1), REST_TOKEN, dtype=torch.long, device=device)

    for step in range(chord.shape[1]):
        if step == 0:
            prev_seq = previous
        else:
            prev_seq = torch.as_tensor([[REST_TOKEN] + generated], dtype=torch.long, device=device)
        logits = model(prev_seq, chord[:, : step + 1, :])[:, -1, :] / max(temperature, 1e-6)
        if decoding == "greedy":
            token = int(torch.argmax(logits, dim=-1).item())
        elif decoding == "top_k":
            k = min(max(top_k, 1), logits.shape[-1])
            values, indices = torch.topk(logits, k=k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            token = int(indices.gather(-1, sample).item())
        else:
            raise ValueError(f"Unknown decoding mode: {decoding}")
        generated.append(token)

    return np.asarray(generated, dtype=np.int64)


def melody_metrics(melody: np.ndarray, chord: Optional[np.ndarray] = None) -> Dict:
    """Compute distributional and harmonic metrics for melody token arrays."""
    tokens = np.asarray(melody).reshape(-1)
    note_mask = (tokens >= 0) & (tokens <= 127)
    notes = tokens[note_mask]
    total_steps = max(tokens.size, 1)

    pitch_class = np.zeros(12, dtype=np.float64)
    if notes.size:
        counts = np.bincount(notes % 12, minlength=12).astype(np.float64)
        pitch_class = counts / counts.sum()

    intervals = np.zeros(25, dtype=np.float64)
    if notes.size >= 2:
        diffs = np.diff(notes).clip(-12, 12) + 12
        interval_counts = np.bincount(diffs, minlength=25).astype(np.float64)
        intervals = interval_counts / interval_counts.sum()

    if chord is not None:
        flat_chord = np.asarray(chord).reshape(-1, CHORD_FEATURE_DIM)
        bar_positions = flat_chord[: tokens.size, 3].astype(np.int64).clip(0, 15)
    else:
        flat_chord = None
        bar_positions = np.arange(tokens.size, dtype=np.int64) % 16

    bar_onsets = np.zeros(16, dtype=np.float64)
    if np.any(note_mask):
        onset_positions = bar_positions[note_mask[: bar_positions.size]]
        bar_counts = np.bincount(onset_positions, minlength=16).astype(np.float64)
        bar_onsets = bar_counts / bar_counts.sum()

    chord_tone_hits = 0
    chord_tone_total = 0
    if flat_chord is not None:
        flat_tokens = np.asarray(melody).reshape(-1)
        for token, feat in zip(flat_tokens, flat_chord):
            if 0 <= token <= 127:
                tones = chord_tones(int(feat[0]), int(feat[1]))
                if tones is not None:
                    chord_tone_total += 1
                    chord_tone_hits += int((int(token) % 12) in tones)

    return {
        "chord_tone_ratio": chord_tone_hits / chord_tone_total if chord_tone_total else None,
        "pitch_class_distribution": pitch_class.tolist(),
        "interval_distribution_-12_to_12": intervals.tolist(),
        "bar_position_onset_distribution": bar_onsets.tolist(),
        "rest_ratio": float(np.mean(tokens == REST_TOKEN)),
        "onset_density": float(np.mean(note_mask)),
        "num_onsets": int(note_mask.sum()),
        "num_steps": int(total_steps),
    }


def summarize_metrics(metrics: Sequence[Dict]) -> Dict:
    """Average scalar metrics and distributions across examples."""
    if not metrics:
        return {}
    summary: Dict[str, object] = {}
    for key in metrics[0].keys():
        values = [m[key] for m in metrics if m.get(key) is not None]
        if not values:
            summary[key] = None
            continue
        first = values[0]
        if isinstance(first, list):
            summary[key] = np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()
        else:
            summary[key] = float(np.mean(values))
    return summary


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two discrete distributions."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = np.maximum(p_arr, eps)
    q_arr = np.maximum(q_arr, eps)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    midpoint = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * np.log(p_arr / midpoint))
    kl_qm = np.sum(q_arr * np.log(q_arr / midpoint))
    return float(0.5 * (kl_pm + kl_qm))


def format_chord_feature(feature: Sequence[int]) -> str:
    root = ROOT_NAMES[int(feature[0])] if 0 <= int(feature[0]) < len(ROOT_NAMES) else "UNK"
    quality = QUALITY_NAMES[int(feature[1])] if 0 <= int(feature[1]) < len(QUALITY_NAMES) else "other"
    if quality == "none":
        return "N"
    return f"{root}:{quality}"


def save_checkpoint(path: str, model: nn.Module, model_name: str, model_config: Dict, epoch: int, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_config": model_config,
            "epoch": epoch,
            "val_loss": val_loss,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))
