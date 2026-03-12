# Wi-Fi CSI Human Activity Recognition for ESP32-Oriented Edge Systems

Compact, reproducible pipeline for human activity recognition from Wi-Fi Channel State Information using Doppler spectrograms and a small CNN evaluated with leave-one-person-out testing.

## Overview

- Converts raw CSI `.mat` recordings into Doppler spectrogram traces
- Trains a compact CNN on fixed-size Doppler tensors
- Evaluates with leave-one-person-out splits across subjects
- Produces plots and a PDF report for analysis

## Repository Layout

- `compute_doppler.py`: raw CSI preprocessing and Doppler extraction
- `train.py`: LOPO training and evaluation
- `plot_doppler.py`: exploratory Doppler visualizations and summaries
- `HAR_20MHz/`: raw dataset directory; download it separately and keep only the 20 MHz subset used here
- `doppler_traces/`: precomputed Doppler tensors
- `docs/out/report.pdf`: compiled report artifact

## Dataset

The expected inputs are MATLAB `.mat` recordings stored under `HAR_20MHz/`, each with a `CSI` field of shape `(T, 64)`.

The data comes from EHUNAM, a Wi-Fi CSI-based dataset for human and machine sensing (de Armas et al., 2025). The activity classes are `empty`, `stand`, `walk`, and `jump`.

Example filenames:

- `MC1_01A_1_HAR_a_J_#_#_01.mat` for a jump activity by person A
- `MC1_02A_1_E_#_#_#_#_01.mat` for an empty room recording.

The preprocessing stage keeps the 52 data subcarriers and resamples each recording into a fixed-size Doppler representation for the CNN.

## Requirements

Python 3.10+ is recommended.

Required packages:

```bash
python3 -m pip install numpy scipy matplotlib scikit-learn torch
```

Optional package:

```bash
python3 -m pip install onnx
```

## Quick Start

### 1. Precompute Doppler traces

```bash
python3 compute_doppler.py \
  --raw-root HAR_20MHz \
  --out-root doppler_traces
```

This generates `doppler_traces/<class>/*.pkl` files with:

- `doppler`
- `doppler_uniform`
- `freq_hz`
- metadata such as config, person, setup, and activity

### 2. Run LOPO training and evaluation

This trains a 4516-parameter 2D CNN on the precomputed Doppler tensors with leave-one-person-out splits.

```bash
python3 train.py \
  --trace-root doppler_traces \
  --out plots/lopo_cnn.pdf
```

Generated outputs:

- `plots/lopo_cnn.pdf`
- model checkpoints under `plots/models/`

### 3. Generate figures

```bash
python3 plot_doppler.py summary \
  --trace-root doppler_traces \
  --out plots/doppler_summary.pdf
```

Other useful plotting modes:

```bash
python3 plot_doppler.py compare \
  --trace-root doppler_traces \
  --config MC1_01A \
  --setup 1 \
  --out plots/doppler_compare_01A_s1.pdf

python3 plot_doppler.py classes \
  --trace-root doppler_traces \
  --out plots/doppler_class_means.pdf

python3 plot_doppler.py panel \
  doppler_traces/walk/MC1_01A_1_HAR_a_W_\#_\#_01.pkl \
  doppler_traces/jump/MC1_01A_1_HAR_a_J_\#_\#_01.pkl \
  --shared-norm \
  --out plots/doppler_panel.pdf
```

## Results

The results below come from the current `HAR_20MHz` subset, which contains only the 20 MHz EHUNAM campaigns `MC1_01A` and `MC1_02A`.

### LOPO Summary

- Mean fold accuracy (segment): `0.803`
- Mean fold accuracy (trace): `0.853`
- Overall accuracy (segment): `0.803`
- Overall accuracy (trace): `0.853`

### Segment-Level Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| empty |     0.955 |  0.902 |    0.928 |     420 |
| stand |     0.862 |  0.800 |    0.830 |     210 |
| walk  |     0.644 |  0.800 |    0.713 |     210 |
| jump  |     0.650 |  0.610 |    0.629 |     210 |
| macro avg | 0.777 | 0.778 | 0.775 | 1050 |
| weighted avg | 0.813 | 0.803 | 0.806 | 1050 |

Overall segment accuracy: `0.803` on `1050` segments.

### Trace-Level Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| empty |     1.000 |  0.950 |    0.974 |      60 |
| stand |     0.962 |  0.833 |    0.893 |      30 |
| walk  |     0.658 |  0.833 |    0.735 |      30 |
| jump  |     0.724 |  0.700 |    0.712 |      30 |
| macro avg | 0.836 | 0.829 | 0.829 | 150 |
| weighted avg | 0.869 | 0.853 | 0.858 | 150 |

Overall trace accuracy: `0.853` on `150` traces.

## References

- de Armas, E., Diaz, G., Sobron, I., et al.  
  **EHUNAM, a WiFi CSI-based dataset for human and machine sensing.**  
  *Scientific Data*, 12(1), 1950, 2025.  DOI: 10.1038/s41597-025-06238-4
