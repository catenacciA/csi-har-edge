#!/usr/bin/env python3
"""
Compute amplitude-based Doppler spectrograms from HAR_20MHz .mat files.

"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom

SUB_START = 6
SUB_END = 58
EPS = 1e-12

CONFIG_PARAMS = {
    "MC1_01A": {
        "fs_hz": 10.0,
        "window_len": 50,
        "hop": 1,
        "n_fft": 128,
    },
    "MC1_02A": {
        "fs_hz": 20.0,
        "window_len": 100,
        "hop": 2,
        "n_fft": 256,
    },
}

FREQ_MAX = 5.0
TARGET_FREQ = 65
TARGET_TIME = 200

FILE_RE = re.compile(r"^(MC1_0[12]A)" r"_(\d+)" r"_(E|HAR)" r"_([a-e#])" r"_([JSWF#])")
ACTIVITY_MAP = {"J": "jump", "S": "stand", "W": "walk", "F": "fall"}


def parse_filename(stem: str) -> dict[str, str]:
    m = FILE_RE.search(stem)
    if m is None:
        raise ValueError(f"Cannot parse filename: {stem}")

    config = m.group(1)
    setup = m.group(2)
    scenario = m.group(3)
    person = m.group(4)
    act_code = m.group(5)

    if scenario == "E":
        activity = "empty"
        person = "none"
    else:
        activity = ACTIVITY_MAP.get(act_code)
        if activity is None:
            raise ValueError(f"Unknown activity code '{act_code}' in {stem}")

    return {
        "config": config,
        "setup": setup,
        "person": person,
        "activity": activity,
        "stem": stem,
    }


def load_csi(mat_path: Path) -> np.ndarray:
    mat = sio.loadmat(str(mat_path))
    csi = np.squeeze(mat["CSI"])
    if csi.ndim != 2 or csi.shape[1] != 64:
        raise ValueError(f"{mat_path.name}: unexpected shape {csi.shape}")
    return csi.astype(np.complex128, copy=False)


def compute_doppler(
    csi_raw: np.ndarray,
    fs_hz: float,
    window_len: int,
    hop: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    csi = csi_raw[:, SUB_START:SUB_END]

    good = np.sum(np.abs(csi), axis=1) > 0
    csi = csi[good]
    if csi.shape[0] < window_len + 1:
        raise ValueError(
            f"Too few good packets ({csi.shape[0]}) for window={window_len}"
        )

    amp = np.abs(csi).astype(np.float64)

    win = np.hanning(window_len).reshape(-1, 1)
    n_win = (amp.shape[0] - window_len) // hop + 1
    n_freq = n_fft // 2 + 1

    doppler = np.zeros((n_win, n_freq), dtype=np.float64)
    for i in range(n_win):
        s = i * hop
        chunk = amp[s : s + window_len]
        chunk = chunk - np.mean(chunk, axis=0, keepdims=True)
        chunk = chunk * win
        dft = np.fft.rfft(chunk, n=n_fft, axis=0)
        doppler[i] = np.sum(np.abs(dft) ** 2, axis=1)

    doppler[:, 0] = 0.0
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    return doppler, freq_hz


def crop_to_freq_range(
    doppler: np.ndarray,
    freq_hz: np.ndarray,
    f_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = freq_hz <= f_max + EPS
    return doppler[:, mask], freq_hz[mask]


def resample_to_uniform(
    doppler: np.ndarray,
    target_time: int,
    target_freq: int,
) -> np.ndarray:
    if doppler.shape[0] == 0 or doppler.shape[1] == 0:
        return np.zeros((target_time, target_freq), dtype=np.float64)
    scale_t = target_time / doppler.shape[0]
    scale_f = target_freq / doppler.shape[1]
    return zoom(doppler, (scale_t, scale_f), order=1)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute amplitude-based Doppler spectrograms for HAR_20MHz"
    )
    ap.add_argument(
        "--raw-root", default="HAR_20MHz", help="Directory with raw .mat files"
    )
    ap.add_argument("--out-root", default="doppler_traces", help="Output root by class")
    ap.add_argument("--target-time", type=int, default=TARGET_TIME)
    ap.add_argument("--target-freq", type=int, default=TARGET_FREQ)
    ap.add_argument("--freq-max", type=float, default=FREQ_MAX)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw-root not found: {raw_root}")

    mat_files = sorted(raw_root.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {raw_root}")

    print(f"[INFO] {len(mat_files)} .mat files in {raw_root}")
    print(f"[INFO] Uniform output grid: {args.target_time} x {args.target_freq}")
    print(f"[INFO] Freq range: [0, {args.freq_max}] Hz")
    print("[INFO] Method: amplitude-based body-speed estimation")

    stats = {"ok": 0, "fail": 0, "skip": 0}
    config_counts = {"MC1_01A": 0, "MC1_02A": 0}

    for mat_path in mat_files:
        try:
            meta = parse_filename(mat_path.stem)
        except ValueError:
            print(f"[SKIP] {mat_path.name}")
            stats["skip"] += 1
            continue

        cfg = CONFIG_PARAMS[meta["config"]]
        fs_hz = cfg["fs_hz"]
        window_len = cfg["window_len"]
        hop = cfg["hop"]
        n_fft = cfg["n_fft"]

        cls_dir = out_root / meta["activity"]
        cls_dir.mkdir(parents=True, exist_ok=True)
        out_pkl = cls_dir / f"{mat_path.stem}.pkl"

        try:
            csi = load_csi(mat_path)
            dop_raw, freq_hz = compute_doppler(csi, fs_hz, window_len, hop, n_fft)
            dop_crop, freq_crop = crop_to_freq_range(dop_raw, freq_hz, args.freq_max)
            dop_uniform = resample_to_uniform(
                dop_crop, args.target_time, args.target_freq
            )

            params = {
                "fs_hz": fs_hz,
                "window_len": window_len,
                "hop": hop,
                "n_fft": n_fft,
                "sub_start": SUB_START,
                "sub_end": SUB_END,
                "freq_max": args.freq_max,
                "target_time": args.target_time,
                "target_freq": args.target_freq,
                "method": "amplitude_rfft_dc_notch",
            }

            payload = {
                "doppler": dop_crop,
                "freq_hz": freq_crop,
                "doppler_uniform": dop_uniform,
                "config": meta["config"],
                "person": meta["person"],
                "activity": meta["activity"],
                "setup": meta["setup"],
                "fs_hz": fs_hz,
                "params": params,
                "stem": meta["stem"],
            }
            with open(out_pkl, "wb") as fp:
                pickle.dump(payload, fp)

            config_counts[meta["config"]] += 1
            print(
                f"[OK] {mat_path.name} config={meta['config']} fs={fs_hz:.0f}Hz "
                f"person={meta['person']} raw={dop_raw.shape} crop={dop_crop.shape} "
                f"uniform={dop_uniform.shape} power={float(np.sum(dop_crop)):.2e}"
            )
            stats["ok"] += 1
        except Exception as e:
            print(f"[ERR] {mat_path.name}: {e}")
            stats["fail"] += 1

    print(f"\n[DONE] ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}")
    print(
        f"[DONE] MC1_01A: {config_counts['MC1_01A']} MC1_02A: {config_counts['MC1_02A']}"
    )
    print(f"Output in: {out_root}/")


if __name__ == "__main__":
    main()
