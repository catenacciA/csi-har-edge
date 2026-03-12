#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12
DB_FLOOR = -30.0
DB_CEIL = 0.0
CMAP = "inferno"
FREQ_MAX = 5.0

CLASS_ORDER = ["empty", "stand", "walk", "jump"]
HAR_CLASSES = ["stand", "walk", "jump"]
PERSON_ORDER = ["a", "b", "c", "d", "e"]
CONFIG_ORDER = ["MC1_01A", "MC1_02A"]
SETUP_ORDER = ["1", "2", "3"]

CLASS_LABELS = {
    "empty": "Empty Room",
    "jump": "Jumping",
    "stand": "Standing",
    "walk": "Walking",
}

CLASS_COLORS = {
    "empty": "#78909C",
    "stand": "#1E88E5",
    "walk": "#43A047",
    "jump": "#E53935",
}


def _apply_presentation_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as fp:
        obj = pickle.load(fp)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict payload")
    return obj


def load_all_traces(trace_root: Path) -> list[dict]:
    records = []
    for cls_dir in sorted(trace_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for pkl in sorted(cls_dir.glob("*.pkl")):
            try:
                rec = load_pkl(pkl)
                rec["_path"] = str(pkl)
                records.append(rec)
            except Exception as exc:
                print(f"[WARN] {pkl.name}: {exc}")
    return records


def get_doppler_tensor(rec: dict) -> np.ndarray:
    key = "doppler_uniform" if "doppler_uniform" in rec else "doppler"
    return np.asarray(rec[key], dtype=np.float64)


def get_raw_doppler_tensor(rec: dict) -> np.ndarray:
    return np.asarray(rec["doppler"], dtype=np.float64)


def doppler_to_db(
    dop: np.ndarray,
    db_floor: float = DB_FLOOR,
    db_ceil: float = DB_CEIL,
    global_max: float | None = None,
) -> np.ndarray:
    mx = global_max if global_max is not None else float(np.max(dop))
    if mx < EPS:
        return np.full_like(dop, db_floor)
    db = 10.0 * np.log10(np.maximum(dop / mx, EPS))
    return np.clip(db, db_floor, db_ceil)


def filter_records(
    records: list[dict],
    *,
    config: str | None = None,
    setup: int | None = None,
    activity: str | None = None,
    person: str | None = None,
) -> list[dict]:
    filtered = records
    if config is not None:
        filtered = [rec for rec in filtered if rec.get("config") == config]
    if setup is not None:
        filtered = [rec for rec in filtered if str(rec.get("setup")) == str(setup)]
    if activity is not None:
        filtered = [rec for rec in filtered if rec.get("activity") == activity]
    if person is not None:
        filtered = [rec for rec in filtered if rec.get("person") == person]
    return filtered


def estimate_duration_s(rec: dict, dop: np.ndarray | None = None) -> float:
    dop = get_doppler_tensor(rec) if dop is None else dop
    fs_hz = float(rec.get("fs_hz", 10.0))
    params = rec.get("params", {})
    hop = int(params.get("hop", 1))
    window_len = int(params.get("window_len", 1))
    n_windows = (
        get_raw_doppler_tensor(rec).shape[0] if "doppler" in rec else dop.shape[0]
    )
    if n_windows <= 0:
        return 0.0
    return ((n_windows - 1) * hop + window_len) / fs_hz


def build_time_axis(rec: dict, n_t: int) -> np.ndarray:
    duration_s = estimate_duration_s(rec)
    if n_t <= 1:
        return np.array([0.5 * duration_s], dtype=np.float64)
    return np.linspace(0.0, duration_s, n_t, dtype=np.float64)


def build_freq_axis(rec: dict, n_f: int, *, prefer_raw: bool = False) -> np.ndarray:
    if prefer_raw and "freq_hz" in rec:
        freq_hz = np.asarray(rec["freq_hz"], dtype=np.float64)
        if len(freq_hz) == n_f:
            return freq_hz
    if "freq_hz" in rec:
        freq_hz = np.asarray(rec["freq_hz"], dtype=np.float64)
        if len(freq_hz) == n_f:
            return freq_hz
    return np.linspace(0.0, FREQ_MAX, n_f, dtype=np.float64)


def axis_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    dx = (x[1] - x[0]) if len(x) > 1 else max(x[0], 1.0)
    dy = (y[1] - y[0]) if len(y) > 1 else 0.1
    return [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]


def render_spectrogram(
    ax: plt.Axes,
    rec: dict,
    *,
    tensor_mode: str = "uniform",
    global_max: float | None = None,
    db_min: float = DB_FLOOR,
    db_max: float = DB_CEIL,
) -> matplotlib.image.AxesImage:
    dop = (
        get_raw_doppler_tensor(rec) if tensor_mode == "raw" else get_doppler_tensor(rec)
    )
    db = doppler_to_db(dop, db_floor=db_min, db_ceil=db_max, global_max=global_max)
    time_axis = build_time_axis(rec, dop.shape[0])
    freq_axis = build_freq_axis(rec, dop.shape[1], prefer_raw=(tensor_mode == "raw"))
    return ax.imshow(
        db.T,
        origin="lower",
        aspect="auto",
        cmap=CMAP,
        vmin=db_min,
        vmax=db_max,
        extent=axis_extent(time_axis, freq_axis),
        interpolation="bilinear",
    )


def describe_record(rec: dict) -> str:
    activity = CLASS_LABELS.get(rec.get("activity", "?"), rec.get("activity", "?"))
    return (
        f"{activity} | cfg={rec.get('config', '?')} | setup={rec.get('setup', '?')} | "
        f"person={rec.get('person', '?')}"
    )


def score_distinctiveness(dop: np.ndarray, activity: str) -> float:
    log_dop = np.log1p(dop)
    _, n_f = log_dop.shape
    if activity == "empty":
        return -float(np.std(log_dop))
    if activity == "stand":
        mid = log_dop[:, n_f // 6 : n_f // 3]
        return float(np.mean(mid)) - 0.5 * float(np.std(log_dop))
    if activity == "walk":
        band = log_dop[:, n_f // 8 : n_f // 2]
        return float(np.std(np.mean(band, axis=1))) + float(np.mean(band))
    if activity == "jump":
        hi = log_dop[:, n_f // 3 :]
        return float(np.max(hi)) + float(np.mean(hi))
    return 0.0


def select_representative_records(records: list[dict]) -> dict[str, dict]:
    best: dict[str, tuple[float, dict]] = {}
    for rec in records:
        activity = rec.get("activity", "")
        if activity not in CLASS_ORDER:
            continue
        score = score_distinctiveness(get_doppler_tensor(rec), activity)
        if activity not in best or score > best[activity][0]:
            best[activity] = (score, rec)
    missing = [cls for cls in CLASS_ORDER if cls not in best]
    if missing:
        raise ValueError(f"No traces for classes: {missing}")
    return {cls: best[cls][1] for cls in CLASS_ORDER}


def dominant_frequency_hz(rec: dict) -> float:
    dop = get_raw_doppler_tensor(rec)
    freq_hz = build_freq_axis(rec, dop.shape[1], prefer_raw=True)
    power = np.mean(dop, axis=0)
    return float(freq_hz[int(np.argmax(power))])


def average_log_power(rec: dict) -> float:
    return float(np.mean(np.log1p(get_raw_doppler_tensor(rec))))


def build_class_stacks(
    records: list[dict],
) -> tuple[dict[str, list[np.ndarray]], tuple[int, int]]:
    shape_counts: dict[tuple[int, int], int] = defaultdict(int)
    for rec in records:
        shape_counts[tuple(get_doppler_tensor(rec).shape)] += 1
    target_shape = max(shape_counts.items(), key=lambda item: item[1])[0]

    by_class: dict[str, list[np.ndarray]] = {}
    for cls in CLASS_ORDER:
        tensors = [
            get_doppler_tensor(rec) for rec in records if rec.get("activity") == cls
        ]
        by_class[cls] = [
            tensor for tensor in tensors if tuple(tensor.shape) == target_shape
        ]

    missing = [cls for cls in CLASS_ORDER if not by_class[cls]]
    if missing:
        raise ValueError(
            f"No usable traces for classes {missing} with shape {target_shape}"
        )
    return by_class, target_shape


def save_figure(fig: plt.Figure, out_path: Path, *, dpi: int = 250) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cmd_panel(args: argparse.Namespace) -> None:
    _apply_presentation_style()

    paths = [Path(path) for path in args.doppler_pkls]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    data = [load_pkl(path) for path in paths]
    global_max = None
    if args.shared_norm:
        global_max = max(float(np.max(get_raw_doppler_tensor(rec))) for rec in data)

    n = len(paths)
    if n <= 2:
        nrows, ncols = 1, n
    elif n <= 4:
        nrows, ncols = 2, 2
    elif n <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = (n + 3) // 4, 4

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    last_im = None
    for idx, (path, rec) in enumerate(zip(paths, data)):
        ax = axes[idx // ncols][idx % ncols]
        last_im = render_spectrogram(
            ax,
            rec,
            tensor_mode="raw",
            global_max=global_max,
            db_min=args.db_min,
            db_max=args.db_max,
        )
        ax.set_title(path.stem, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Doppler Frequency [Hz]")
        ax.set_ylim(0, FREQ_MAX)
        ax.text(
            0.02,
            0.98,
            describe_record(rec),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
            },
        )

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), pad=0.02, shrink=0.88)
        cbar.set_label("Power [dB]")

    save_figure(fig, Path(args.out), dpi=220)
    print(f"[OK] Saved panel plot -> {args.out}")


def _draw_compare_figure(
    out_path: Path,
    selected: dict[str, dict],
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 3.6), constrained_layout=True, squeeze=False
    )
    global_max = max(
        float(np.max(get_doppler_tensor(rec))) for rec in selected.values()
    )
    last_im = None
    for ci, cls in enumerate(CLASS_ORDER):
        ax = axes[0][ci]
        rec = selected[cls]
        last_im = render_spectrogram(ax, rec, global_max=global_max)
        ax.set_title(
            CLASS_LABELS.get(cls, cls),
            fontweight="bold",
            color=CLASS_COLORS.get(cls, "black"),
            fontsize=14,
            pad=8,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, FREQ_MAX)
        ax.set_ylabel("Doppler Frequency [Hz]" if ci == 0 else "")
        ax.text(
            0.02,
            0.02,
            f"p={rec.get('person', '?')}  cfg={rec.get('config', '?')}  s={rec.get('setup', '?')}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
            },
        )
    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes[0].tolist(), pad=0.015, shrink=0.92, aspect=25
        )
        cbar.set_label("Power [dB]")
    save_figure(fig, out_path, dpi=dpi)


def cmd_compare(args: argparse.Namespace) -> None:
    _apply_presentation_style()

    records = load_all_traces(Path(args.trace_root))
    records = filter_records(records, config=args.config, setup=args.setup)
    if not records:
        raise ValueError("No traces after applying filters")

    selected = select_representative_records(records)
    out_path = Path(args.out)
    _draw_compare_figure(out_path, selected, dpi=300)
    if args.png:
        png_path = out_path.with_suffix(".png")
        _draw_compare_figure(png_path, selected, dpi=300)
        print(f"[OK] Saved compare PNG -> {png_path}")
    print(f"[OK] Saved compare plot -> {out_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    _apply_presentation_style()

    records = load_all_traces(Path(args.trace_root))
    records = filter_records(records, config=args.config, setup=args.setup)
    if not records:
        raise ValueError("No traces after applying filters")

    counts_by_class = [
        sum(rec.get("activity") == cls for rec in records) for cls in CLASS_ORDER
    ]

    coverage = np.zeros((len(PERSON_ORDER), len(HAR_CLASSES)), dtype=int)
    for pi, person in enumerate(PERSON_ORDER):
        for ci, cls in enumerate(HAR_CLASSES):
            coverage[pi, ci] = sum(
                rec.get("person") == person and rec.get("activity") == cls
                for rec in records
            )

    durations = {cls: [] for cls in CLASS_ORDER}
    peak_freqs = {cls: [] for cls in CLASS_ORDER}
    power_by_class = {cls: [] for cls in CLASS_ORDER}
    setup_config = np.zeros((len(CONFIG_ORDER), len(SETUP_ORDER)), dtype=int)
    for rec in records:
        cls = rec.get("activity")
        durations[cls].append(estimate_duration_s(rec))
        peak_freqs[cls].append(dominant_frequency_hz(rec))
        power_by_class[cls].append(average_log_power(rec))
        cfg = rec.get("config")
        setup = str(rec.get("setup"))
        if cfg in CONFIG_ORDER and setup in SETUP_ORDER:
            setup_config[CONFIG_ORDER.index(cfg), SETUP_ORDER.index(setup)] += 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0][0]
    x = np.arange(len(CLASS_ORDER))
    bars = ax.bar(
        x,
        counts_by_class,
        color=[CLASS_COLORS[cls] for cls in CLASS_ORDER],
        width=0.68,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CLASS_LABELS[cls] for cls in CLASS_ORDER], rotation=15, ha="right"
    )
    ax.set_ylabel("Trace count")
    ax.set_title("Dataset Balance", fontweight="bold")
    for rect, value in zip(bars, counts_by_class):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            value + 0.3,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[0][1]
    im = ax.imshow(coverage, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(HAR_CLASSES)))
    ax.set_xticklabels([CLASS_LABELS[cls] for cls in HAR_CLASSES])
    ax.set_yticks(range(len(PERSON_ORDER)))
    ax.set_yticklabels([f"Person {person.upper()}" for person in PERSON_ORDER])
    ax.set_title("Person-Activity Coverage", fontweight="bold")
    for pi in range(coverage.shape[0]):
        for ci in range(coverage.shape[1]):
            ax.text(
                ci,
                pi,
                str(coverage[pi, ci]),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02, label="Trace count")

    ax = axes[1][0]
    duration_series = [durations[cls] for cls in CLASS_ORDER]
    box = ax.boxplot(
        duration_series,
        patch_artist=True,
        tick_labels=[CLASS_LABELS[cls] for cls in CLASS_ORDER],
    )
    for patch, cls in zip(box["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.45)
    ax.set_ylabel("Estimated duration [s]")
    ax.set_title("Trace Duration by Class", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1][1]
    peak_series = [peak_freqs[cls] for cls in CLASS_ORDER]
    box = ax.boxplot(
        peak_series,
        patch_artist=True,
        tick_labels=[CLASS_LABELS[cls] for cls in CLASS_ORDER],
    )
    for patch, cls in zip(box["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.45)
    ax.set_ylabel("Dominant Doppler frequency [Hz]")
    ax.set_title("Dominant Frequency by Class", fontweight="bold")
    ax.set_ylim(0, FREQ_MAX)
    ax.tick_params(axis="x", rotation=15)

    title_parts = ["Dataset Summary"]
    if args.config is not None:
        title_parts.append(args.config)
    if args.setup is not None:
        title_parts.append(f"setup {args.setup}")
    fig.suptitle(" | ".join(title_parts), fontsize=15, fontweight="bold")

    save_figure(fig, Path(args.out), dpi=260)
    print(f"[OK] Saved summary -> {args.out}")
    print(f"[INFO] Total traces: {len(records)}")
    for cls in CLASS_ORDER:
        cls_records = [rec for rec in records if rec.get("activity") == cls]
        print(
            f"[INFO] {cls:>5s}: n={len(cls_records):3d}  "
            f"duration={np.mean(durations[cls]):.2f}s  peak={np.mean(peak_freqs[cls]):.2f}Hz  "
            f"log-power={np.mean(power_by_class[cls]):.3f}"
        )
    for cfg in CONFIG_ORDER:
        row_idx = CONFIG_ORDER.index(cfg)
        row_counts = ", ".join(
            f"setup {setup}={setup_config[row_idx, si]}"
            for si, setup in enumerate(SETUP_ORDER)
        )
        print(f"[INFO] {cfg}: {row_counts}")


def cmd_classes(args: argparse.Namespace) -> None:
    _apply_presentation_style()

    records = load_all_traces(Path(args.trace_root))
    records = filter_records(records, config=args.config, setup=args.setup)
    if not records:
        raise ValueError("No traces after applying filters")

    by_class, _ = build_class_stacks(records)
    all_tensors = [tensor for cls in CLASS_ORDER for tensor in by_class[cls]]
    global_max = max(float(np.max(tensor)) for tensor in all_tensors)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    last_im = None
    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx // 2][idx % 2]
        mean_map = np.mean(np.stack(by_class[cls], axis=0), axis=0)
        db = doppler_to_db(mean_map, global_max=global_max)
        time_axis = np.linspace(0.0, 1.0, mean_map.shape[0], dtype=np.float64)
        freq_axis = np.linspace(0.0, FREQ_MAX, mean_map.shape[1], dtype=np.float64)
        last_im = ax.imshow(
            db.T,
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            vmin=DB_FLOOR,
            vmax=DB_CEIL,
            extent=axis_extent(time_axis, freq_axis),
            interpolation="bilinear",
        )
        ax.set_title(
            f"{CLASS_LABELS[cls]} (n={len(by_class[cls])})",
            fontweight="bold",
            color=CLASS_COLORS[cls],
        )
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("Doppler Frequency [Hz]")
        ax.set_ylim(0, FREQ_MAX)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.87, pad=0.02)
        cbar.set_label("Power [dB]")

    title_parts = ["Class Mean Doppler Maps"]
    if args.config is not None:
        title_parts.append(args.config)
    if args.setup is not None:
        title_parts.append(f"setup {args.setup}")
    fig.suptitle(" | ".join(title_parts), fontsize=15, fontweight="bold")

    save_figure(fig, Path(args.out), dpi=280)
    print(f"[OK] Saved class means -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot useful Doppler summaries from precomputed traces"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_panel = sub.add_parser(
        "panel", help="Inspect one or more specific Doppler traces"
    )
    sp_panel.add_argument("doppler_pkls", nargs="+")
    sp_panel.add_argument("--out", default="plots/doppler_panel.pdf")
    sp_panel.add_argument("--db-min", type=float, default=DB_FLOOR)
    sp_panel.add_argument("--db-max", type=float, default=DB_CEIL)
    sp_panel.add_argument("--shared-norm", action="store_true")

    sp_compare = sub.add_parser(
        "compare", help="Compare representative traces across classes"
    )
    sp_compare.add_argument("--trace-root", default="doppler_traces")
    sp_compare.add_argument("--out", default="plots/doppler_compare.pdf")
    sp_compare.add_argument("--config", default=None, choices=CONFIG_ORDER)
    sp_compare.add_argument("--setup", type=int, default=None, choices=[1, 2, 3])
    sp_compare.add_argument("--png", action="store_true", help="Also save a PNG copy")

    sp_summary = sub.add_parser(
        "summary", help="Create a compact dataset overview dashboard"
    )
    sp_summary.add_argument("--trace-root", default="doppler_traces")
    sp_summary.add_argument("--out", default="plots/doppler_summary.pdf")
    sp_summary.add_argument("--config", default=None, choices=CONFIG_ORDER)
    sp_summary.add_argument("--setup", type=int, default=None, choices=[1, 2, 3])

    sp_classes = sub.add_parser("classes", help="Plot class-average Doppler maps")
    sp_classes.add_argument("--trace-root", default="doppler_traces")
    sp_classes.add_argument("--out", default="plots/doppler_class_means.pdf")
    sp_classes.add_argument("--config", default=None, choices=CONFIG_ORDER)
    sp_classes.add_argument("--setup", type=int, default=None, choices=[1, 2, 3])

    sp_showcase = sub.add_parser("showcase", help="Alias for compare")
    sp_showcase.add_argument("--trace-root", default="doppler_traces")
    sp_showcase.add_argument("--out", default="plots/showcase.pdf")
    sp_showcase.add_argument("--config", default=None, choices=CONFIG_ORDER)
    sp_showcase.add_argument("--setup", type=int, default=None, choices=[1, 2, 3])
    sp_showcase.add_argument("--png", action="store_true", help="Also save a PNG copy")

    args = ap.parse_args()

    if args.mode == "panel":
        cmd_panel(args)
    elif args.mode in {"compare", "showcase"}:
        cmd_compare(args)
    elif args.mode == "summary":
        cmd_summary(args)
    elif args.mode == "classes":
        cmd_classes(args)


if __name__ == "__main__":
    main()
