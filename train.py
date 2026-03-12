#!/usr/bin/env python3
"""
LOPO evaluation of a compact CNN on Doppler spectrogram traces.

The script loads Doppler traces, segments them into windows, runs LOPO
cross-validation, and reports both classification results and a rough
ESP32-S3 resource estimate.

To reduce leakage, folds are assigned at trace level and empty recordings
from the same session are kept in the same fold.
"""

from __future__ import annotations

import argparse
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Segmentation parameters ─────────────────────────────────────────
SEG_WINDOW = 50  # time steps per segment (out of 200 uniform)
SEG_HOP = 25  # hop between segments -> 50% overlap

# ─── Classes ──────────────────────────────────────────────────────────
CLASS_ORDER = ["empty", "stand", "walk", "jump"]
CLASS2IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)

PERSON_ORDER = ["a", "b", "c", "d", "e"]
N_FOLDS = len(PERSON_ORDER)
REPETITION_RE = re.compile(r"_(\d+)$")
BASE_SEED = 42


def configure_reproducibility(seed: int) -> None:
    """Configure deterministic CPU training for repeatable metrics."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


# =====================================================================
#  Data loading, fold assignment, segmentation
# =====================================================================


def load_all_traces(trace_root: Path) -> list[dict]:
    """Load every .pkl under trace_root/{class}/."""
    records = []
    for cls_dir in sorted(trace_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for pkl in sorted(cls_dir.glob("*.pkl")):
            with open(pkl, "rb") as fp:
                d = pickle.load(fp)
            if isinstance(d, dict):
                d["_path"] = str(pkl)
                records.append(d)
    return records


def assign_folds(records: list[dict]) -> list[dict]:
    """
    Assign a fold index to each trace.

    Non-empty traces use the person ID.
    Empty traces are grouped by session metadata and assigned by group.
    """
    person2fold = {p: i for i, p in enumerate(PERSON_ORDER)}

    # keep empty traces from the same session in the same fold
    empty_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)

    for r in records:
        if r["activity"] == "empty":
            stem = str(r.get("stem", ""))
            rep_match = REPETITION_RE.search(stem)
            repetition = int(rep_match.group(1)) if rep_match else -1
            setup_raw = str(r.get("setup", ""))
            setup = int(setup_raw) if setup_raw.isdigit() else -1
            key = (
                str(r.get("config", "NA")),
                setup,
                repetition,
            )
            empty_groups[key].append(r)
        else:
            person = str(r.get("person", "")).lower()
            if person not in person2fold:
                raise ValueError(
                    f"Unknown person '{person}' in record {r.get('stem', r.get('_path'))}"
                )
            r["fold"] = person2fold[person]

    empty_keys = sorted(empty_groups.keys(), key=lambda k: (k[0], k[1], k[2]))
    for idx, key in enumerate(empty_keys):
        fold = idx % N_FOLDS
        for r in empty_groups[key]:
            r["fold"] = fold

    return records


def segment_one_trace(
    dop: np.ndarray,
    window: int,
    hop: int,
) -> np.ndarray:
    """
    Segment one Doppler trace into windows and apply per-trace normalisation.
    """
    dop = np.asarray(dop, dtype=np.float32)
    dop = np.log1p(dop)
    mu, sigma = dop.mean(), dop.std()
    if sigma > 1e-8:
        dop = (dop - mu) / sigma

    n_t, n_f = dop.shape
    segs = []
    for s in range(0, n_t - window + 1, hop):
        seg = dop[s : s + window, :, np.newaxis]  # (W, F, 1)
        segs.append(seg)
    if not segs:
        return np.empty((0, window, n_f, 1), dtype=np.float32)
    return np.array(segs, dtype=np.float32)


def build_fold_arrays(
    records: list[dict],
    fold_test: int | None,
    window: int = SEG_WINDOW,
    hop: int = SEG_HOP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build train/test arrays for one fold.

    If fold_test is None, all traces are returned in the training split.
    """
    train_X, train_y, test_X, test_y = [], [], [], []
    train_tid, test_tid = [], []

    for rec in records:
        activity = rec.get("activity")
        if activity not in CLASS2IDX:
            continue
        cls_idx = CLASS2IDX[activity]
        dop_key = "doppler_uniform" if "doppler_uniform" in rec else "doppler"
        segs = segment_one_trace(np.asarray(rec[dop_key]), window, hop)
        if segs.shape[0] == 0:
            continue
        labels = np.full(segs.shape[0], cls_idx, dtype=np.int64)
        trace_id = str(rec.get("stem", rec.get("_path", "unknown_trace")))
        trace_ids = np.full(segs.shape[0], trace_id, dtype=object)

        if fold_test is not None and rec["fold"] == fold_test:
            test_X.append(segs)
            test_y.append(labels)
            test_tid.append(trace_ids)
        else:
            train_X.append(segs)
            train_y.append(labels)
            train_tid.append(trace_ids)

    X_tr = np.concatenate(train_X) if train_X else np.empty((0,))
    y_tr = np.concatenate(train_y) if train_y else np.empty((0,), dtype=np.int64)
    tid_tr = np.concatenate(train_tid) if train_tid else np.empty((0,), dtype=object)
    X_te = np.concatenate(test_X) if test_X else np.empty((0,))
    y_te = np.concatenate(test_y) if test_y else np.empty((0,), dtype=np.int64)
    tid_te = np.concatenate(test_tid) if test_tid else np.empty((0,), dtype=object)
    return X_tr, y_tr, tid_tr, X_te, y_te, tid_te


# =====================================================================
#  Data augmentation
# =====================================================================


def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply simple augmentation to a batch of Doppler segments.
    """
    B, T, F, C = X.shape
    X_aug = X.copy()

    for i in range(B):
        # random time shift
        if rng.random() < 0.5:
            shift = rng.integers(-5, 6)
            X_aug[i, :, :, 0] = np.roll(X_aug[i, :, :, 0], shift, axis=0)

        # random Gaussian noise
        if rng.random() < 0.5:
            sigma = rng.uniform(0.0, 0.1)
            X_aug[i] += rng.normal(0, sigma, X_aug[i].shape).astype(np.float32)

        # random frequency masking
        if rng.random() < 0.3:
            n_mask = rng.integers(1, 4)
            mask_start = rng.integers(0, max(1, F - n_mask))
            X_aug[i, :, mask_start : mask_start + n_mask, :] = 0.0

    return X_aug


# =====================================================================
#  CNN model
# =====================================================================


def build_cnn(input_shape: tuple[int, ...], num_classes: int):
    """
    Architecture  (input: 1 x 50 x 65):
      Conv2d(1->16, 3x3)           -> BN -> ReLU -> MaxPool(2x2)   25x32x16
      DepthwiseSep(16->32, 3x3)    -> BN -> ReLU -> MaxPool(2x2)   12x16x32
      DepthwiseSep(32->48, 3x3)    -> BN -> ReLU -> MaxPool(2x2)     6x8x48
      AvgPool(6x8) -> flatten -> 48
      Linear(48->32) -> ReLU -> Dropout(0.25) -> Linear(32->4)
    """
    import torch
    import torch.nn as nn

    class DepthwiseSeparable(nn.Module):

        def __init__(self, in_ch, out_ch, kernel=3, padding=1):
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False
            )
            self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            return self.relu(x)

    class DopplerCNN(nn.Module):

        def __init__(self):
            super().__init__()
            # initial convolution block
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            # depthwise-separable feature blocks
            self.block2 = nn.Sequential(
                DepthwiseSeparable(16, 32),
                nn.MaxPool2d(2),
            )
            self.block3 = nn.Sequential(
                DepthwiseSeparable(32, 48),
                nn.MaxPool2d(2),
            )
            # fixed pooling after the three downsampling stages
            self.pool = nn.AvgPool2d((6, 8))
            self.classifier = nn.Sequential(
                nn.Linear(48, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            # x: (B, T, F, 1) -> (B, 1, T, F)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x)
            x = x.reshape(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = DopplerCNN()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


# =====================================================================
#  ESP32 resource estimation
# =====================================================================


def estimate_esp32_resources(model, input_shape: tuple[int, ...]) -> dict:
    """
    Estimate model size, MAC count, and activation memory for ESP32-S3.
    """
    import torch

    n_params = sum(p.numel() for p in model.parameters())

    # Count MACs
    mac_count = [0]
    hooks = []

    def conv_mac_hook(module, inp, out):
        if isinstance(module, torch.nn.Conv2d):
            out_h, out_w = out.shape[2], out.shape[3]
            k = module.kernel_size[0] * module.kernel_size[1]
            in_ch = module.in_channels // module.groups
            mac_count[0] += out_h * out_w * module.out_channels * k * in_ch
        elif isinstance(module, torch.nn.Linear):
            mac_count[0] += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(conv_mac_hook))

    dummy = torch.zeros(1, *input_shape)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()

    # Peak activation memory
    act_sizes = []
    act_hooks = []

    def act_hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            act_sizes.append(out.nelement() * 4)

    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.Conv2d,
                torch.nn.BatchNorm2d,
                torch.nn.ReLU,
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.Linear,
            ),
        ):
            act_hooks.append(m.register_forward_hook(act_hook))

    act_sizes.clear()
    with torch.no_grad():
        model(dummy)
    for h in act_hooks:
        h.remove()

    peak_act_bytes = max(act_sizes) if act_sizes else 0

    esp32_freq_hz = 240_000_000
    cycles_per_mac = 10
    est_ms = (mac_count[0] * cycles_per_mac) / esp32_freq_hz * 1000

    return {
        "params": n_params,
        "flash_fp32_kb": n_params * 4 / 1024,
        "flash_fp16_kb": n_params * 2 / 1024,
        "flash_int8_kb": n_params * 1 / 1024,
        "total_macs": mac_count[0],
        "peak_activation_kb": peak_act_bytes / 1024,
        "est_inference_ms": est_ms,
        "input_size_kb": np.prod(input_shape) * 4 / 1024,
    }


# =====================================================================
#  Training loop
# =====================================================================


def split_train_val_by_trace(
    X: np.ndarray,
    y: np.ndarray,
    trace_ids: np.ndarray,
    rng: np.random.Generator,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data by trace ID, not by segment.
    """
    if len(X) != len(y) or len(y) != len(trace_ids):
        raise ValueError("X, y, trace_ids must have the same length")

    if len(X) == 0 or val_ratio <= 0:
        X_val = np.empty((0,) + X.shape[1:], dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
        return X, y, X_val, y_val

    unique_traces, first_idx = np.unique(trace_ids, return_index=True)
    trace_labels = y[first_idx]

    val_trace_set: set[str] = set()
    for cls_idx in range(NUM_CLASSES):
        cls_traces = unique_traces[trace_labels == cls_idx]
        n_cls = len(cls_traces)
        if n_cls <= 1:
            continue
        n_val = int(round(val_ratio * n_cls))
        n_val = min(max(n_val, 1), n_cls - 1)
        picked = rng.choice(cls_traces, size=n_val, replace=False)
        val_trace_set.update(str(x) for x in picked.tolist())

    # fallback if the class-wise split gives no validation traces
    if not val_trace_set and len(unique_traces) > 1:
        n_val = int(round(val_ratio * len(unique_traces)))
        n_val = min(max(n_val, 1), len(unique_traces) - 1)
        picked = rng.choice(unique_traces, size=n_val, replace=False)
        val_trace_set.update(str(x) for x in picked.tolist())

    if not val_trace_set:
        X_val = np.empty((0,) + X.shape[1:], dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
        return X, y, X_val, y_val

    trace_ids_str = np.asarray(trace_ids, dtype=str)
    val_mask = np.isin(trace_ids_str, np.array(sorted(val_trace_set), dtype=str))
    train_mask = ~val_mask

    if not np.any(val_mask) or not np.any(train_mask):
        X_val = np.empty((0,) + X.shape[1:], dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
        return X, y, X_val, y_val

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def aggregate_trace_predictions(
    y_true_seg: np.ndarray,
    logits_seg: np.ndarray,
    trace_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate segment predictions at trace level using mean logits.
    """
    if len(y_true_seg) != len(logits_seg) or len(y_true_seg) != len(trace_ids):
        raise ValueError("y_true_seg, logits_seg, trace_ids must have the same length")

    logits_by_trace: dict[str, list[np.ndarray]] = defaultdict(list)
    true_by_trace: dict[str, int] = {}

    for i, tid_raw in enumerate(trace_ids):
        tid = str(tid_raw)
        logits_by_trace[tid].append(logits_seg[i])
        yi = int(y_true_seg[i])
        if tid in true_by_trace and true_by_trace[tid] != yi:
            raise ValueError(f"Trace {tid} has inconsistent true labels")
        true_by_trace[tid] = yi

    if not logits_by_trace:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    y_true_trace, y_pred_trace = [], []
    for tid in sorted(logits_by_trace):
        trace_logits = np.stack(logits_by_trace[tid], axis=0)
        mean_logits = np.mean(trace_logits, axis=0)
        y_true_trace.append(true_by_trace[tid])
        y_pred_trace.append(int(np.argmax(mean_logits)))

    return np.asarray(y_true_trace, dtype=np.int64), np.asarray(
        y_pred_trace, dtype=np.int64
    )


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_trace_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    augment: bool = True,
    seed: int = BASE_SEED,
    val_ratio: float = 0.2,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, object]:
    """Train CNN on one LOPO fold and evaluate it on test segments."""
    model = fit_model(
        X_train,
        y_train,
        train_trace_ids,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        augment=augment,
        seed=seed,
        val_ratio=val_ratio,
    )

    y_true, y_pred, logits_seg = evaluate_model(model, X_test, y_test)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return acc, y_true, y_pred, logits_seg, model


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_trace_ids: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    augment: bool = True,
    seed: int = BASE_SEED,
    val_ratio: float = 0.2,
):
    """Train the CNN on trace-grouped Doppler segments."""
    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    if len(X_train) == 0:
        raise ValueError("X_train is empty")

    configure_reproducibility(seed)
    rng = np.random.default_rng(seed)
    X_fit, y_fit, X_val, y_val = split_train_val_by_trace(
        X_train, y_train, train_trace_ids, rng=rng, val_ratio=val_ratio
    )
    has_val = len(X_val) > 0
    if has_val:
        print(f"  [VAL] train segments={len(X_fit)} val segments={len(X_val)}")
    else:
        print(
            "  [VAL] no validation split available, using train loss for early stopping"
        )

    model, n_params = build_cnn(X_train.shape[1:], NUM_CLASSES)
    model = model.to(device)
    print(f"[CNN] Params: {n_params:,}")

    # compensate for class imbalance
    class_counts = np.bincount(y_fit, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    def eval_loss_acc(X_eval: np.ndarray, y_eval: np.ndarray) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X_eval), 256):
                xb = torch.tensor(X_eval[i : i + 256], device=device)
                yb = torch.tensor(y_eval[i : i + 256], device=device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        return total_loss / max(total, 1), total_correct / max(total, 1)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(X_fit))
        X_shuf = X_fit[perm]
        y_shuf = y_fit[perm]

        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X_shuf), batch_size):
            xb = X_shuf[i : i + batch_size]
            yb = y_shuf[i : i + batch_size]

            if augment:
                xb = augment_batch(xb, rng)

            xb_t = torch.tensor(xb, device=device)
            yb_t = torch.tensor(yb, device=device)

            optimiser.zero_grad()
            logits = model(xb_t)
            loss = criterion(logits, yb_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb_t).sum().item()
            total += len(yb)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if has_val:
            val_loss, val_acc = eval_loss_acc(X_val, y_val)
            monitor_loss = val_loss
        else:
            val_loss, val_acc = train_loss, train_acc
            monitor_loss = train_loss

        if monitor_loss < best_loss - 1e-4:
            best_loss = monitor_loss
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if has_val:
                print(
                    f"    epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                    f"train_acc={train_acc:.3f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                )
            else:
                print(
                    f"    epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                    f"train_acc={train_acc:.3f}"
                )

        if patience_counter >= patience:
            print(f"    early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run segment-level inference for a trained model."""
    import torch

    if len(X_eval) == 0:
        y_empty = np.empty((0,), dtype=np.int64)
        logits_empty = np.empty((0, NUM_CLASSES), dtype=np.float32)
        return y_empty, y_empty, logits_empty

    model.eval()
    all_preds, all_true, all_logits = [], [], []
    with torch.no_grad():
        for i in range(0, len(X_eval), 256):
            xb = torch.tensor(X_eval[i : i + 256], device="cpu")
            logits = model(xb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(y_eval[i : i + 256])
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    logits_seg = np.concatenate(all_logits)
    return y_true, y_pred, logits_seg


# =====================================================================
#  Plot helpers
# =====================================================================


def _plot_per_fold_accuracy(
    fold_results: list[dict],
    mean_fold_acc_seg: float,
    mean_fold_acc_trace: float,
    colors: list[str],
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)

    _draw_per_fold_accuracy(
        ax, fold_results, mean_fold_acc_seg, mean_fold_acc_trace, colors
    )
    return fig, ax


def _draw_per_fold_accuracy(
    ax: plt.Axes,
    fold_results: list[dict],
    mean_fold_acc_seg: float,
    mean_fold_acc_trace: float,
    colors: list[str],
) -> None:
    persons_lbl = [f["person"].upper() for f in fold_results]
    seg_accs = [f["acc_seg"] for f in fold_results]
    trace_accs = [f["acc_trace"] for f in fold_results]

    x = np.arange(len(persons_lbl))
    bar_w = 0.38

    ax.bar(
        x - bar_w / 2,
        seg_accs,
        width=bar_w,
        label="Segment",
        color=colors[: len(persons_lbl)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + bar_w / 2,
        trace_accs,
        width=bar_w,
        label="Trace",
        color="#90CAF9",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        mean_fold_acc_seg,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean seg = {mean_fold_acc_seg:.3f}",
    )
    ax.axhline(
        mean_fold_acc_trace,
        color="#1E88E5",
        linestyle=":",
        linewidth=1.2,
        label=f"Mean trace = {mean_fold_acc_trace:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(persons_lbl)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Test person (LOPO)")
    ax.set_title("Per-fold segment/trace accuracy", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    for i, (vs, vt) in enumerate(zip(seg_accs, trace_accs)):
        ax.text(
            i - bar_w / 2,
            vs + 0.02,
            f"{vs:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            i + bar_w / 2,
            vt + 0.02,
            f"{vt:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )


def _draw_per_class_accuracy_by_fold(
    ax: plt.Axes,
    fold_results: list[dict],
    class_order: list[str],
    colors: list[str],
) -> None:
    per_class_per_fold = _compute_per_class_accuracy_by_fold(fold_results, class_order)

    x_pos = np.arange(len(class_order))
    width = 0.15

    for fi, fold in enumerate(fold_results):
        vals = [per_class_per_fold[cls][fi] for cls in class_order]
        ax.bar(
            x_pos + fi * width,
            vals,
            width,
            label=f"P{fold['person'].upper()}",
            color=colors[fi],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(class_order)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy by fold", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.1)


def _draw_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    class_order: list[str],
    title: str,
    tick_labels: list[str] | None = None,
    text_fmt: str = "{norm:.2f}\n({count})",
    text_size: int = 8,
):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = tick_labels if tick_labels is not None else class_order

    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(
        labels,
        rotation=45 if tick_labels is None else 0,
        ha="right" if tick_labels is None else "center",
    )
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")

    for i in range(len(class_order)):
        for j in range(len(class_order)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                text_fmt.format(norm=cm_norm[i, j], count=cm[i, j]),
                ha="center",
                va="center",
                fontsize=text_size,
                color=color,
            )

    return im


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_order: list[str],
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    im = _draw_confusion_matrix(
        ax=ax,
        cm=cm,
        class_order=class_order,
        title="Normalised confusion matrix",
    )

    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig, ax


def _compute_per_class_accuracy_by_fold(
    fold_results: list[dict],
    class_order: list[str],
) -> dict[str, list[float]]:
    per_class_per_fold: dict[str, list[float]] = defaultdict(list)
    for fold in fold_results:
        y_true = fold["y_true_seg"]
        y_pred = fold["y_pred_seg"]
        for class_idx, class_name in enumerate(class_order):
            mask = y_true == class_idx
            if np.any(mask):
                per_class_per_fold[class_name].append(
                    float(np.mean(y_true[mask] == y_pred[mask]))
                )
            else:
                per_class_per_fold[class_name].append(0.0)
    return per_class_per_fold


def _format_esp32_resource_summary(
    resources: dict,
    model_ram: float,
    esp32_sram_kb: float,
    include_status: bool = False,
) -> str:
    lines = [
        "ESP32-S3 Deployment Estimate",
        "_" * 35,
        f"Parameters:        {resources['params']:,}",
        f"Flash (int8):      {resources['flash_int8_kb']:.1f} KB",
        f"Flash (float16):   {resources['flash_fp16_kb']:.1f} KB",
        f"Input buffer:      {resources['input_size_kb']:.1f} KB",
        f"Peak activation:   {resources['peak_activation_kb']:.1f} KB",
        f"Total MACs:        {resources['total_macs']:,}",
        f"Est. inference:    ~{resources['est_inference_ms']:.0f} ms",
        "_" * 35,
        f"RAM budget:        {model_ram:.1f} KB / {esp32_sram_kb} KB",
    ]
    if include_status:
        status = (
            "Fits in SRAM budget"
            if model_ram < esp32_sram_kb * 0.5
            else "Prefer PSRAM headroom"
        )
        lines.append(f"Status:            {status}")
    lines.extend(
        [
            "",
            "Live pipeline:",
            "  CSI -> amplitude -> STFT -> CNN",
            f"  Latency: ~{resources['est_inference_ms']:.0f} ms + ~50 ms features",
            f"  Total:   ~{resources['est_inference_ms'] + 50:.0f} ms",
        ]
    )
    return "\n".join(lines)


def _plot_per_class_accuracy_by_fold(
    fold_results: list[dict],
    class_order: list[str],
    colors: list[str],
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    _draw_per_class_accuracy_by_fold(ax, fold_results, class_order, colors)

    return fig, ax


def _draw_esp32_resource_summary(
    ax: plt.Axes,
    resources: dict,
    model_ram: float,
    esp32_sram_kb: float,
    include_status: bool = False,
) -> None:
    esp_text = _format_esp32_resource_summary(
        resources=resources,
        model_ram=model_ram,
        esp32_sram_kb=esp32_sram_kb,
        include_status=include_status,
    )
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        esp_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )
    ax.set_title("ESP32 resource summary", fontweight="bold")


def _plot_esp32_resource_summary(
    resources: dict,
    model_ram: float,
    esp32_sram_kb: float,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    _draw_esp32_resource_summary(ax, resources, model_ram, esp32_sram_kb)
    return fig, ax


def _plot_per_fold_confusion_matrices(
    fold_results: list[dict],
    class_order: list[str],
) -> tuple[plt.Figure, list[plt.Axes]]:
    from sklearn.metrics import confusion_matrix

    n_folds = len(fold_results)
    fig, axes = plt.subplots(
        1, n_folds, figsize=(3.2 * n_folds, 3.2), constrained_layout=True
    )
    if n_folds == 1:
        axes = [axes]

    for fi, f in enumerate(fold_results):
        ax = axes[fi]
        yt = f["y_true_seg"]
        yp = f["y_pred_seg"]
        cm_fold = confusion_matrix(yt, yp, labels=range(len(class_order)))
        _draw_confusion_matrix(
            ax=ax,
            cm=cm_fold,
            class_order=class_order,
            title=f"P{f['person'].upper()} S:{f['acc_seg']:.2f} T:{f['acc_trace']:.2f}",
            tick_labels=[c[0].upper() for c in class_order],
            text_fmt="{count}",
            text_size=7,
        )
        ax.title.set_fontsize(9)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle("Per-fold confusion matrices", fontsize=12, fontweight="bold")
    return fig, axes


def save_plots(
    out_path: Path,
    plot_layout: str,
    fold_results: list[dict],
    mean_fold_acc_seg: float,
    mean_fold_acc_trace: float,
    overall_acc_seg: float,
    overall_acc_trace: float,
    cm: np.ndarray,
    class_order: list[str],
    resources: dict,
    model_ram: float,
    esp32_sram_kb: float,
) -> None:
    colors = ["#E91E63", "#9C27B0", "#3F51B5", "#009688", "#FF9800"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if plot_layout == "combined":
        from sklearn.metrics import confusion_matrix

        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        # (a) Per-fold accuracy
        ax = fig.add_subplot(gs[0, 0])
        _draw_per_fold_accuracy(
            ax, fold_results, mean_fold_acc_seg, mean_fold_acc_trace, colors
        )

        # (b) Confusion matrix
        ax = fig.add_subplot(gs[0, 1])
        im = _draw_confusion_matrix(
            ax=ax,
            cm=cm,
            class_order=class_order,
            title="Normalised confusion matrix",
        )
        fig.colorbar(im, ax=ax, shrink=0.8)

        # (c) Per-class accuracy across folds
        ax = fig.add_subplot(gs[0, 2])
        _draw_per_class_accuracy_by_fold(ax, fold_results, class_order, colors)

        # (d) ESP32 resource summary
        ax = fig.add_subplot(gs[1, 0])
        _draw_esp32_resource_summary(
            ax,
            resources=resources,
            model_ram=model_ram,
            esp32_sram_kb=esp32_sram_kb,
            include_status=True,
        )

        # (e) Per-fold confusion matrices
        ax_big = fig.add_subplot(gs[1, 1:])
        ax_big.axis("off")
        inner_gs = gs[1, 1:].subgridspec(1, len(fold_results), wspace=0.3)
        for fi, f in enumerate(fold_results):
            ax_inner = fig.add_subplot(inner_gs[0, fi])
            yt = f["y_true_seg"]
            yp = f["y_pred_seg"]
            cm_fold = confusion_matrix(yt, yp, labels=range(len(class_order)))
            _draw_confusion_matrix(
                ax=ax_inner,
                cm=cm_fold,
                class_order=class_order,
                title=f"P{f['person'].upper()} S:{f['acc_seg']:.2f} T:{f['acc_trace']:.2f}",
                tick_labels=[c[0].upper() for c in class_order],
                text_fmt="{count}",
                text_size=7,
            )
            ax_inner.title.set_fontsize(9)
            ax_inner.tick_params(axis="both", labelsize=7)

        fig.suptitle(
            f"LOPO CNN (ESP32-ready)  --  Seg={overall_acc_seg:.3f} "
            f"Trace={overall_acc_trace:.3f}  --  {resources['params']:,} params",
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] Saved results -> {out_path}")
        return

    # split mode
    stem = out_path.stem
    out_dir = out_path.parent

    fig, _ = _plot_per_fold_accuracy(
        fold_results=fold_results,
        mean_fold_acc_seg=mean_fold_acc_seg,
        mean_fold_acc_trace=mean_fold_acc_trace,
        colors=colors,
    )
    p = out_dir / f"{stem}_per_fold_accuracy.pdf"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {p}")

    fig, _ = _plot_confusion_matrix(cm=cm, class_order=class_order)
    p = out_dir / f"{stem}_confusion_matrix.pdf"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {p}")

    fig, _ = _plot_per_class_accuracy_by_fold(
        fold_results=fold_results,
        class_order=class_order,
        colors=colors,
    )
    p = out_dir / f"{stem}_per_class_accuracy_by_fold.pdf"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {p}")

    fig, _ = _plot_esp32_resource_summary(
        resources=resources,
        model_ram=model_ram,
        esp32_sram_kb=esp32_sram_kb,
    )
    p = out_dir / f"{stem}_esp32_resource_summary.pdf"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {p}")

    fig, _ = _plot_per_fold_confusion_matrices(
        fold_results=fold_results,
        class_order=class_order,
    )
    p = out_dir / f"{stem}_per_fold_confusion_matrices.pdf"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {p}")


# =====================================================================
#  Main
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LOPO CNN on Doppler spectrograms with ESP32-S3 resource estimation"
    )
    ap.add_argument("--trace-root", default="doppler_traces")
    ap.add_argument("--out", default="plots/lopo_cnn.pdf")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seg-hop", type=int, default=SEG_HOP)
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Trace-level validation ratio for early stopping",
    )
    ap.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation"
    )
    ap.add_argument(
        "--plot-layout",
        choices=["combined", "split"],
        default="combined",
        help="Save plots into one multi-panel PDF ('combined') or as separate PDFs ('split').",
    )
    args = ap.parse_args()

    print("=" * 65)
    print("  LOPO CNN Evaluation on Doppler Spectrograms")
    print("=" * 65)

    # -- Load & assign folds -------------------------------------------
    records = load_all_traces(Path(args.trace_root))
    records = assign_folds(records)
    print(f"[DATA] {len(records)} traces loaded")

    # Show fold distribution
    fold_dist = Counter((r["fold"], r["activity"]) for r in records)
    for fi in range(N_FOLDS):
        label = PERSON_ORDER[fi].upper()
        counts = {c: fold_dist.get((fi, c), 0) for c in CLASS_ORDER}
        print(f"  Fold {fi} (Person {label}): {dict(counts)}")

    # -- Leakage audit -------------------------------------------------
    print("\n[AUDIT] Checking for data leakage...")
    for fi in range(N_FOLDS):
        test_stems = {r["stem"] for r in records if r["fold"] == fi}
        train_stems = {r["stem"] for r in records if r["fold"] != fi}
        overlap = test_stems & train_stems
        status = "CLEAN" if not overlap else f"LEAK DETECTED: {overlap}"
        print(
            f"  Fold {fi}: test={len(test_stems)} train={len(train_stems)} -> {status}"
        )

    # -- ESP32 resource report -----------------------------------------
    print("\n[ESP32] Resource estimation...")
    import torch

    configure_reproducibility(BASE_SEED)
    model_check, n_params = build_cnn((SEG_WINDOW, 65, 1), NUM_CLASSES)
    resources = estimate_esp32_resources(model_check, (SEG_WINDOW, 65, 1))

    print(f"  Parameters:          {resources['params']:>10,}")
    print(f"  Flash (int8):        {resources['flash_int8_kb']:>10.1f} KB")
    print(f"  Flash (float16):     {resources['flash_fp16_kb']:>10.1f} KB")
    print(f"  Flash (float32):     {resources['flash_fp32_kb']:>10.1f} KB")
    print(f"  Input buffer:        {resources['input_size_kb']:>10.1f} KB")
    print(f"  Peak activation:     {resources['peak_activation_kb']:>10.1f} KB")
    print(f"  Total MACs:          {resources['total_macs']:>10,}")
    print(f"  Est. inference (ESP32-S3, int8): ~{resources['est_inference_ms']:.0f} ms")

    esp32_sram_kb = 512
    esp32_psram_kb = 8192
    model_ram = resources["flash_int8_kb"] + resources["peak_activation_kb"]
    print(f"\n  Total RAM needed:    {model_ram:.1f} KB")
    print(
        f"  ESP32-S3 SRAM:       {esp32_sram_kb} KB  "
        f"{'OK' if model_ram < esp32_sram_kb * 0.5 else 'Prefer PSRAM headroom'}"
    )

    print("\n  [ESP32] Live feature computation budget:")
    print(f"    STFT window:       50 frames x 52 subcarriers")
    print(f"    FFT per window:    52x rfft(50->128) = 52 x 128-pt FFTs")
    print(f"    Segment buffer:    50 x 65 x 4 = {50*65*4/1024:.1f} KB")
    print(f"    Total pipeline:    Feature compute + CNN inference")
    print(f"    Estimated budget fits an ESP32-S3 pipeline at 10 Hz CSI rate")

    # -- LOPO evaluation -----------------------------------------------
    print("\n" + "=" * 65)
    print("  LOPO Cross-Validation (4 classes: empty, stand, walk, jump)")
    print("=" * 65)

    fold_results = []
    all_y_true_seg, all_y_pred_seg = [], []
    all_y_true_trace, all_y_pred_trace = [], []

    for fi in range(N_FOLDS):
        person = PERSON_ORDER[fi]
        print(f"\n{'_' * 55}")
        print(f"FOLD {fi+1}/{N_FOLDS}:  test = Person {person.upper()}")
        print(f"{'_' * 55}")

        X_train, y_train, train_trace_ids, X_test, y_test, test_trace_ids = (
            build_fold_arrays(records, fi, SEG_WINDOW, args.seg_hop)
        )

        class_train = dict(Counter(y_train.tolist()))
        class_test = dict(Counter(y_test.tolist()))
        print(f"  train: {len(X_train)}  classes={class_train}")
        print(f"  test:  {len(X_test)}   classes={class_test}")

        if len(X_test) == 0:
            print(f"  [SKIP] No test data")
            continue

        seg_acc, yt_seg, yp_seg, logits_seg, _ = train_one_fold(
            X_train,
            y_train,
            train_trace_ids,
            X_test,
            y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=not args.no_augment,
            seed=BASE_SEED + fi,
            val_ratio=args.val_ratio,
        )

        yt_trace, yp_trace = aggregate_trace_predictions(
            yt_seg, logits_seg, test_trace_ids
        )
        trace_acc = float(np.mean(yt_trace == yp_trace)) if len(yt_trace) else 0.0

        fold_results.append(
            {
                "person": person,
                "acc_seg": seg_acc,
                "acc_trace": trace_acc,
                "n_test_seg": len(yt_seg),
                "n_test_trace": len(yt_trace),
                "class_test": class_test,
                "y_true_seg": yt_seg,
                "y_pred_seg": yp_seg,
            }
        )
        all_y_true_seg.append(yt_seg)
        all_y_pred_seg.append(yp_seg)
        all_y_true_trace.append(yt_trace)
        all_y_pred_trace.append(yp_trace)

        print(
            f"  => Person {person.upper()} segment_acc: {seg_acc:.3f} "
            f"({int(seg_acc * len(yt_seg))}/{len(yt_seg)})  "
            f"trace_acc: {trace_acc:.3f} "
            f"({int(trace_acc * len(yt_trace))}/{len(yt_trace)})"
        )

    if not fold_results:
        print("[ERR] No folds completed")
        return

    # -- Overall results -----------------------------------------------
    all_y_true_seg = np.concatenate(all_y_true_seg)
    all_y_pred_seg = np.concatenate(all_y_pred_seg)
    all_y_true_trace = np.concatenate(all_y_true_trace)
    all_y_pred_trace = np.concatenate(all_y_pred_trace)

    overall_acc_seg = float(np.mean(all_y_true_seg == all_y_pred_seg))
    overall_acc_trace = float(np.mean(all_y_true_trace == all_y_pred_trace))
    mean_fold_acc_seg = np.mean([f["acc_seg"] for f in fold_results])
    mean_fold_acc_trace = np.mean([f["acc_trace"] for f in fold_results])

    print(f"\n{'=' * 65}")
    print(f"  LOPO RESULTS  (all 4 classes including empty)")
    print(f"{'=' * 65}")
    for f in fold_results:
        print(
            f"  Person {f['person'].upper()}:  "
            f"seg={f['acc_seg']:.3f} (n={f['n_test_seg']})  "
            f"trace={f['acc_trace']:.3f} (n={f['n_test_trace']})"
        )
    print(f"  {'_' * 50}")
    print(f"  Mean fold accuracy (segment): {mean_fold_acc_seg:.3f}")
    print(f"  Mean fold accuracy (trace):   {mean_fold_acc_trace:.3f}")
    print(f"  Overall accuracy (segment):   {overall_acc_seg:.3f}")
    print(f"  Overall accuracy (trace):     {overall_acc_trace:.3f}")

    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(all_y_true_seg, all_y_pred_seg, labels=range(NUM_CLASSES))
    print(f"\n  Classification report (segment-level):")
    print(
        classification_report(
            all_y_true_seg,
            all_y_pred_seg,
            target_names=CLASS_ORDER,
            digits=3,
            zero_division=0,
        )
    )
    print(f"\n  Classification report (trace-level):")
    print(
        classification_report(
            all_y_true_trace,
            all_y_pred_trace,
            target_names=CLASS_ORDER,
            digits=3,
            zero_division=0,
        )
    )

    # -- Plot ----------------------------------------------------------

    out_path = Path(args.out)
    save_plots(
        out_path=out_path,
        plot_layout=args.plot_layout,
        fold_results=fold_results,
        mean_fold_acc_seg=mean_fold_acc_seg,
        mean_fold_acc_trace=mean_fold_acc_trace,
        overall_acc_seg=overall_acc_seg,
        overall_acc_trace=overall_acc_trace,
        cm=cm,
        class_order=CLASS_ORDER,
        resources=resources,
        model_ram=model_ram,
        esp32_sram_kb=esp32_sram_kb,
    )

    print("\n[FINAL] Training final model on all traces...")
    X_all, y_all, all_trace_ids, _, _, _ = build_fold_arrays(
        records,
        fold_test=None,
        window=SEG_WINDOW,
        hop=args.seg_hop,
    )
    final_model = fit_model(
        X_all,
        y_all,
        all_trace_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        seed=BASE_SEED + N_FOLDS,
        val_ratio=0.0,
    )

    # -- Export model --------------------------------------------------
    if final_model is not None:
        model_dir = out_path.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        pt_path = model_dir / "doppler_cnn_esp32.pt"
        torch.save(final_model.state_dict(), pt_path)

        try:
            onnx_path = model_dir / "doppler_cnn_esp32.onnx"
            dummy = torch.zeros(1, SEG_WINDOW, 65, 1)
            final_model.eval()
            torch.onnx.export(
                final_model,
                dummy,
                str(onnx_path),
                input_names=["doppler_input"],
                output_names=["class_logits"],
                dynamic_axes={"doppler_input": {0: "batch"}},
                opset_version=13,
            )
            print(f"[OK] ONNX model -> {onnx_path}")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

        print(f"[OK] PyTorch model -> {pt_path}")
        print(f"\n[EXPORT] Suggested deployment path for ESP32-S3:")
        print(f"  1. Export the trained model")
        print(f"  2. Convert and quantise it to int8")
        print(f"  3. Integrate it with TFLite Micro / ESP-IDF")
        print(f"  4. Run inference on Doppler segments from the CSI pipeline")


if __name__ == "__main__":
    main()
