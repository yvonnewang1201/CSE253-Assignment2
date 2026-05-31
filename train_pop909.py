"""Train POP909 symbolic melody generation models."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pop909_utils import HOLD_TOKEN, MELODY_VOCAB_SIZE, REST_TOKEN, MelodyDataset, build_model, perplexity, save_checkpoint, sequence_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: str, split: str):
    chord_path = os.path.join(data_dir, f"{split}_chord.npy")
    melody_path = os.path.join(data_dir, f"{split}_melody.npy")
    if not os.path.exists(chord_path) or not os.path.exists(melody_path):
        raise FileNotFoundError(
            f"Missing processed split '{split}'. Run preprocess_pop909.py first. "
            f"Expected {chord_path} and {melody_path}."
        )
    return np.load(chord_path), np.load(melody_path)


def model_config_from_args(args) -> Dict:
    if args.model == "unconditional_gru":
        return {
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.layers,
            "dropout": args.dropout,
        }
    if args.model == "chord_gru":
        return {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.layers,
            "dropout": args.dropout,
        }
    return {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "max_len": args.seq_len,
    }


def resolve_loss_weights(args):
    """Use fair defaults unless the user explicitly sets loss weights."""
    if (
        args.onset_loss_weight is not None
        and args.rest_loss_weight is not None
        and args.hold_loss_weight is not None
    ):
        return args.onset_loss_weight, args.rest_loss_weight, args.hold_loss_weight

    if args.model == "unconditional_gru":
        return 1.0, 1.0, 1.0
    return 2.0, 0.9, 0.9


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)

    train_chord, train_melody = load_split(args.data_dir, "train")
    val_chord, val_melody = load_split(args.data_dir, "val")
    if len(train_melody) == 0:
        raise RuntimeError("Training split is empty. Check POP909 preprocessing warnings.")
    if len(val_melody) == 0:
        raise RuntimeError("Validation split is empty. Use more songs or adjust preprocessing split.")

    args.seq_len = int(train_melody.shape[1])
    train_dataset = MelodyDataset(train_chord, train_melody)
    val_dataset = MelodyDataset(val_chord, val_melody)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    config = model_config_from_args(args)
    model = build_model(args.model, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    onset_weight, rest_weight, hold_weight = resolve_loss_weights(args)
    weights = torch.ones(MELODY_VOCAB_SIZE, dtype=torch.float32, device=device)
    weights[:128] *= onset_weight
    weights[REST_TOKEN] *= rest_weight
    weights[HOLD_TOKEN] *= hold_weight
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, f"{args.model}_best.pt")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Model config: {config}")
    print(f"Loss weights: onset={onset_weight}, rest={rest_weight}, hold={hold_weight}")
    print(f"Train windows: {len(train_dataset)}, val windows: {len(val_dataset)}")

    epochs_without_improvement = 0
    last_epoch = 0
    val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for previous, chord, target in train_loader:
            previous = previous.to(device)
            chord = chord.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(previous, chord)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = sequence_loss(model, val_loader, device, criterion)
        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} ppl {perplexity(train_loss):.2f} | "
            f"val loss {val_loss:.4f} ppl {perplexity(val_loss):.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args.model, config, epoch, val_loss)
            print(f"  saved best checkpoint -> {best_path}")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"  early stopping: no validation improvement for "
                    f"{args.early_stop_patience} epochs"
                )
                break

    final_path = os.path.join(args.save_dir, f"{args.model}_last.pt")
    save_checkpoint(final_path, model, args.model, config, last_epoch, val_loss)
    print(f"Training complete. Best val loss: {best_val:.4f}. Last checkpoint: {final_path}")


def train_all_default(args) -> None:
    """Train all supported models when --model is omitted."""
    models = ["unconditional_gru", "chord_gru", "chord_transformer"]
    print("No --model supplied; training all default POP909 models:")
    print("  " + ", ".join(models))
    for model_name in models:
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        if model_name == "chord_transformer":
            model_args.hidden_dim = 128
            model_args.layers = 2
        if model_name == "unconditional_gru":
            model_args.onset_loss_weight = 1.0
            model_args.rest_loss_weight = 1.0
            model_args.hold_loss_weight = 1.0
        else:
            model_args.onset_loss_weight = 2.0
            model_args.rest_loss_weight = 0.9
            model_args.hold_loss_weight = 0.9
        print("\n" + "=" * 72)
        print(f"Training {model_name}")
        print("=" * 72)
        train(model_args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train POP909 melody generation models.")
    parser.add_argument(
        "--model",
        choices=["unconditional_gru", "chord_gru", "chord_transformer"],
        default=None,
        help="Model to train. If omitted, trains all three models with the shared defaults.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", default="processed_pop909")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--onset_loss_weight", type=float, default=None)
    parser.add_argument("--rest_loss_weight", type=float, default=None)
    parser.add_argument("--hold_loss_weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=253)
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.set_defaults(seq_len=64)
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if parsed_args.model is None:
        train_all_default(parsed_args)
    else:
        train(parsed_args)
