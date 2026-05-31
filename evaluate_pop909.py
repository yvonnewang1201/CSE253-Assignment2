"""Evaluate POP909 melody generation with distributional and harmonic metrics."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pop909_utils import (
    MelodyDataset,
    generate_tokens,
    js_divergence,
    load_checkpoint,
    melody_metrics,
    perplexity,
    sequence_loss,
    summarize_metrics,
)


def load_split(data_dir: str, split: str):
    chord = np.load(os.path.join(data_dir, f"{split}_chord.npy"))
    melody = np.load(os.path.join(data_dir, f"{split}_melody.npy"))
    return chord, melody


def discover_checkpoints(args) -> List[str]:
    checkpoints: List[str] = []
    if args.checkpoint:
        checkpoints.extend(args.checkpoint)
    if args.unconditional_checkpoint:
        checkpoints.append(args.unconditional_checkpoint)
    if args.chord_checkpoint:
        checkpoints.append(args.chord_checkpoint)
    if args.transformer_checkpoint:
        checkpoints.append(args.transformer_checkpoint)

    default_names = ["unconditional_gru_best.pt", "chord_gru_best.pt", "chord_transformer_best.pt"]
    for name in default_names:
        path = os.path.join(args.save_dir, name)
        if os.path.exists(path) and path not in checkpoints:
            checkpoints.append(path)
    seen = set()
    unique = []
    for path in checkpoints:
        normalized = os.path.normcase(os.path.abspath(os.path.normpath(path)))
        if normalized not in seen:
            unique.append(path)
            seen.add(normalized)
    return unique


def add_distribution_distances(model_result: Dict, real_summary: Dict) -> None:
    generated = model_result["generated"]
    generated["pitch_class_js_divergence"] = js_divergence(
        generated["pitch_class_distribution"],
        real_summary["pitch_class_distribution"],
    )
    generated["interval_js_divergence"] = js_divergence(
        generated["interval_distribution_-12_to_12"],
        real_summary["interval_distribution_-12_to_12"],
    )
    generated["bar_position_onset_js_divergence"] = js_divergence(
        generated["bar_position_onset_distribution"],
        real_summary["bar_position_onset_distribution"],
    )


def evaluate_checkpoint(path: str, data_dir: str, device: torch.device, batch_size: int, num_samples: int,
                        decoding: str, top_k: int, real_summary: Dict) -> Dict:
    model, checkpoint = load_checkpoint(path, device)
    val_chord, val_melody = load_split(data_dir, "val")
    test_chord, test_melody = load_split(data_dir, "test")

    criterion = nn.CrossEntropyLoss()
    val_loader = DataLoader(MelodyDataset(val_chord, val_melody), batch_size=batch_size, shuffle=False)
    val_loss = sequence_loss(model, val_loader, device, criterion)

    limit = min(num_samples, len(test_chord))
    generated_metrics = []
    generated_tokens = []
    for idx in range(limit):
        tokens = generate_tokens(model, test_chord[idx], device, decoding=decoding, top_k=top_k)
        generated_tokens.append(tokens)
        generated_metrics.append(melody_metrics(tokens, test_chord[idx]))

    result = {
        "checkpoint": path,
        "model_name": checkpoint.get("model_name", "unknown"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_loss": val_loss,
        "validation_perplexity": perplexity(val_loss),
        "generated": summarize_metrics(generated_metrics),
        "num_generated_examples": limit,
    }
    add_distribution_distances(result, real_summary)
    return result


def evaluate(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_chord, test_melody = load_split(args.data_dir, "test")
    if len(test_melody) == 0:
        raise RuntimeError("Test split is empty. Preprocess more POP909 songs before evaluation.")

    real_limit = min(args.num_samples, len(test_melody))
    real_summary = summarize_metrics(
        [melody_metrics(test_melody[idx], test_chord[idx]) for idx in range(real_limit)]
    )

    results = {
        "real_melody": real_summary,
        "num_real_examples": real_limit,
        "models": [],
    }

    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found. Pass --checkpoint or train models into --save_dir first."
        )

    for path in checkpoints:
        if not os.path.exists(path):
            print(f"[WARN] Missing checkpoint, skipping: {path}")
            continue
        print(f"Evaluating {path}")
        results["models"].append(
            evaluate_checkpoint(
                path,
                args.data_dir,
                device,
                args.batch_size,
                args.num_samples,
                args.decoding,
                args.top_k,
                real_summary,
            )
        )

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "pop909_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSummary")
    print(f"Real onset density: {real_summary.get('onset_density'):.4f}")
    print(f"Real rest ratio: {real_summary.get('rest_ratio'):.4f}")
    print(f"Real chord-tone ratio: {real_summary.get('chord_tone_ratio')}")
    for model_result in results["models"]:
        metrics = model_result["generated"]
        print(
            f"{model_result['model_name']}: val loss {model_result['validation_loss']:.4f}, "
            f"ppl {model_result['validation_perplexity']:.2f}, "
            f"chord-tone {metrics.get('chord_tone_ratio')}, "
            f"rest {metrics.get('rest_ratio'):.4f}, onset {metrics.get('onset_density'):.4f}, "
            f"pc-js {metrics.get('pitch_class_js_divergence'):.4f}, "
            f"int-js {metrics.get('interval_js_divergence'):.4f}, "
            f"bar-js {metrics.get('bar_position_onset_js_divergence'):.4f}"
        )
    print(f"Full report: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate POP909 generated melody metrics.")
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint to evaluate. Can repeat.")
    parser.add_argument("--unconditional_checkpoint", default=None)
    parser.add_argument("--chord_checkpoint", default=None)
    parser.add_argument("--transformer_checkpoint", default=None)
    parser.add_argument("--data_dir", default="processed_pop909")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="evaluation_pop909")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--decoding", choices=["greedy", "top_k"], default="top_k")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
