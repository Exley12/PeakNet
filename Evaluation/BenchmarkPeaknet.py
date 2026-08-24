# Imports
import sys
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.signal import savgol_filter, butter, filtfilt, wiener, find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

# Add the project root to the path so peaknet can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import peaknet


"""
Benchmark script for evaluating peaknet against other signal processing filters.
Current filters included are Savitzky-Golay, Butterworth low-pass, Gaussian smoothing, and Wiener filtering.
Loads clean masks from specified dataset folder and adds Gaussian noise
Testing on 5 levels of noise
Summary metrics plotted: MSE, F1, SNR, runtime
Also plotting confusion matrices for each filter at each noise level
"""

# Config

# Which dataset folder to load spectra and masks from
DATASET_DIR = Path(__file__).resolve().parent.parent / "Data" / "dataset_small"

# How many spectra to benchmark (None = use every spectrum in the dataset)
N = None

# Which spectrum index (0 to N-1) to save example plots for
PLOT_SPECTRUM = 9

# Fixed random seed so the noise added on top of each spectrum is reproducible
SEED = 88

# The five noise levels to test — sigma is the standard deviation of the Gaussian noise added
NOISE_LEVELS = {
    "noise_verylow":  0.02,
    "noise_low":      0.05,
    "noise_medium":   0.1,
    "noise_high":     0.2,
    "noise_veryhigh": 0.40,
}

# How many bins away a predicted peak centre can be from a true peak and still count as a match
PEAK_MATCH_TOL = 5

# Where to save all benchmark outputs
OUT_DIR = Path(__file__).resolve().parent / "benchmark"

# Plots go into per-noise-level subfolders inside here
PLOTS_DIR = OUT_DIR / "plots"


# Filters

# Savitzky-Golay: fits a polynomial over a sliding window to smooth the signal
def apply_savgol(x):
    return savgol_filter(x, 11, 3)

# Butterworth low-pass: attenuates high-frequency noise above a cutoff frequency
def apply_butter(x):
    b, a = butter(4, 0.1)
    # filtfilt needs padlen < len(x); shrink it for very short spectra
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, padlen=max(padlen, 0))

# Gaussian: convolves the signal with a Gaussian kernel to smooth it
def apply_gaussian(x):
    return gaussian_filter1d(x, sigma=3)

# Wiener: adaptive filter that estimates and removes noise based on local statistics
def apply_wiener(x):
    return wiener(x, 11)

# Peaknet: neural network trained to recover the clean peak mask from a noisy spectrum
def apply_peaknet(x):
    return peaknet(x)

# Dictionary of all filters so they can be looped over consistently
FILTERS = {
    "savgol":   apply_savgol,
    "butter":   apply_butter,
    "gaussian": apply_gaussian,
    "wiener":   apply_wiener,
    "peaknet":  apply_peaknet,
}

# Colours used when plotting each filter's output
FILTER_COLORS = {
    "savgol":   "#e07b39",
    "butter":   "#5b8dd9",
    "gaussian": "#7cba6d",
    "wiener":   "#b05ec4",
    "peaknet":  "#e84040",
}


# Dataset loading

def load_masks(dataset_dir):
    # Pick the masks file, falling back to the peaks-specific name
    masks_path = dataset_dir / "masks.npy"
    if not masks_path.exists():
        masks_path = dataset_dir / "peaks_masks.npy"

    # Masks are ragged (variable length), so they load as an object array
    return np.load(masks_path, allow_pickle=True)

# Peak extraction
def extract_peaks(signal, min_height=0.1):
    # Find all local maxima above the minimum height threshold
    indices, _ = find_peaks(signal, height=min_height)

    # Peak heights at each detected index
    heights = signal[indices]

    # FWHM (rel_height=0.5) via linear interpolation between the samples
    # straddling the half-height crossing, relative to each peak's prominence
    widths, _, _, _ = peak_widths(signal, indices, rel_height=0.5)

    peaks = list(zip(indices.astype(int).tolist(), heights.astype(float).tolist(), widths.tolist()))

    return peaks

# Peak matching
def match_peaks(pred_peaks, true_peaks, tol=PEAK_MATCH_TOL):
    se_heights = []
    se_fwhms   = []
    used = set()
    tp   = 0

    # For each true peak, find the closest predicted peak within tolerance
    for (tc, th, tf) in true_peaks:
        best_dist, best_i = None, None
        for i, (pc, ph, pf) in enumerate(pred_peaks):
            if i in used:
                continue
            d = abs(pc - tc)
            if d <= tol and (best_dist is None or d < best_dist):
                best_dist, best_i = d, i

        # If a match was found, record the squared errors in height and FWHM
        if best_i is not None:
            pc, ph, pf = pred_peaks[best_i]
            se_heights.append((ph - th) ** 2)
            se_fwhms.append((pf - tf) ** 2)
            used.add(best_i)
            tp += 1

    # Average the squared errors — nan if there were no matched peaks
    MSE_peak_height = float(np.mean(se_heights)) if se_heights else np.nan
    MSE_fwhm        = float(np.mean(se_fwhms))   if se_fwhms  else np.nan

    # Squared difference between predicted and true peak count
    MSE_peak_count = (len(pred_peaks) - len(true_peaks)) ** 2

    # Precision: fraction of predicted peaks that matched a true peak
    precision = tp / len(pred_peaks) if pred_peaks else (1.0 if not true_peaks else 0.0)

    # Recall: fraction of true peaks that were found
    recall = tp / len(true_peaks) if true_peaks else (1.0 if not pred_peaks else 0.0)

    # F1: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # False positives: predicted peaks with no matching true peak
    fp = len(pred_peaks) - tp

    # False negatives: true peaks that were not found
    fn = len(true_peaks) - tp

    return MSE_peak_height, MSE_fwhm, MSE_peak_count, f1, tp, fp, fn

# Main

def main():
    # Record when the benchmark started so total runtime can be reported at the end
    benchmark_start = time.perf_counter()

    # Create one output subfolder per noise level for the example plots
    for noise_label in NOISE_LEVELS:
        os.makedirs(PLOTS_DIR / noise_label, exist_ok=True)

    # Seed the random number generator for reproducibility
    rng = np.random.default_rng(SEED)

    # Load the target masks from disk — these double as the clean signal
    masks = load_masks(DATASET_DIR)

    # Benchmark every spectrum in the dataset unless N caps it lower
    n_spectra = len(masks) if N is None else min(N, len(masks))

    # Collect one row of metrics per spectrum × noise level × filter
    rows = []

    # Accumulate TP, FP, FN counts per filter × noise level for the confusion matrices
    cm_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Track how many example plots have been saved
    plot_count = 0

    for i in range(n_spectra):
        # Use the mask itself as the clean signal — it is exactly zero except at the
        # peaks, so the noise sweep is the only source of corruption applied to it
        mask  = masks[i].astype(np.float32)
        clean = mask
        L     = len(clean)

        # Extract the ground-truth peaks from the clean mask
        true_peaks = extract_peaks(mask)

        for noise_label, sigma in NOISE_LEVELS.items():
            # Add Gaussian noise to the clean spectrum and clip to the training range
            noisy = clean + rng.normal(0, sigma, L).astype(np.float32)
            noisy = np.clip(noisy, -0.2, 1.0).astype(np.float32)

            # Save a reference plot showing just the raw noisy input vs the mask
            if i == PLOT_SPECTRUM:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(noisy, color="lightgrey", linewidth=1,   label="Noisy input")
                ax.plot(mask,  color="black",     linewidth=1,   label="Mask", linestyle="--", alpha=0.9)
                # Mark each true peak centre with a vertical dotted line
                for (tc, th, tf) in true_peaks:
                    ax.axvline(tc, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
                ax.set_title(f"spectrum {i} — raw — {noise_label}")
                ax.set_xlabel("Bin")
                ax.set_ylabel("Amplitude")
                ax.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / noise_label / f"s{i:04d}_raw.png", dpi=100)
                plt.close()
                plot_count += 1

            for filter_name, filter_fn in FILTERS.items():
                # Time just the filter call so overhead is not included
                t0         = time.perf_counter()
                denoised   = filter_fn(noisy)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Extract peaks from the filtered output and compare to the true peaks
                pred_peaks = extract_peaks(denoised)
                mse_h, mse_f, mse_c, f1, tp, fp, fn = match_peaks(pred_peaks, true_peaks)

                # Accumulate confusion matrix counts across all spectra
                cm_counts[(filter_name, noise_label)]["tp"] += tp
                cm_counts[(filter_name, noise_label)]["fp"] += fp
                cm_counts[(filter_name, noise_label)]["fn"] += fn

                # MSE between the filtered output and the clean mask
                mse_signal   = float(np.mean((denoised - mask) ** 2))

                # Signal-to-noise ratio in dB — undefined (nan) when signal or residual power is zero
                signal_power = float(np.mean(mask ** 2))
                noise_power  = float(np.mean((denoised - mask) ** 2))
                if signal_power > 0 and noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                else:
                    snr = np.nan

                # Store all metrics for this spectrum × noise × filter combination
                rows.append({
                    "spectrum_id":     i,
                    "filter":          filter_name,
                    "noise_level":     noise_label,
                    "MSE_peak_height": mse_h,
                    "MSE_fwhm":        mse_f,
                    "MSE_peak_count":  mse_c,
                    "F1":              f1,
                    "MSE_signal":      mse_signal,
                    "SNR_dB":          snr,
                    "time_ms":         elapsed_ms,
                })

                # Only save filter plots for the chosen spectrum index
                if i != PLOT_SPECTRUM:
                    continue

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(noisy,    color="lightgrey",            linewidth=1,   label="Noisy input")
                ax.plot(mask,     color="black",                linewidth=1,   label="Mask", linestyle="--", alpha=0.6)
                ax.plot(denoised, color=FILTER_COLORS[filter_name], linewidth=1.5, label=filter_name)
                # Mark each true peak centre with a vertical dotted line
                for (tc, th, tf) in true_peaks:
                    ax.axvline(tc, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
                # Show key metrics in a box in the top-left corner
                mse_h_str = f"{mse_h:.3f}" if not np.isnan(mse_h) else "nan"
                mse_f_str = f"{mse_f:.1f}"  if not np.isnan(mse_f) else "nan"
                txt = f"MSE_peak_height={mse_h_str}  MSE_fwhm={mse_f_str}  MSE_peak_count={mse_c}"
                ax.text(0.02, 0.96, txt, transform=ax.transAxes, fontsize=8,
                        verticalalignment="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                ax.set_title(f"spectrum {i} — {filter_name} — {noise_label}")
                ax.set_xlabel("Bin")
                ax.set_ylabel("Amplitude")
                ax.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / noise_label / f"s{i:04d}_{filter_name}.png", dpi=100)
                plt.close()
                plot_count += 1

    print(f"[info] Saved {plot_count} example plots to {PLOTS_DIR}")

    # Save raw results

    # Save every individual row to a CSV for detailed analysis
    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT_DIR / "results_all.csv", index=False)

    # Build summary (averaged over all spectra)

    # Keep the noise levels in the order they were defined rather than alphabetical
    noise_order = list(NOISE_LEVELS.keys())

    summary_df = (
        all_df
        .groupby(["filter", "noise_level"])[
            ["MSE_peak_height", "MSE_fwhm", "MSE_peak_count", "F1", "MSE_signal", "SNR_dB", "time_ms"]
        ]
        .mean()
        .reset_index()
    )

    # Re-apply the original noise level ordering so rows sort correctly
    summary_df["noise_level"] = pd.Categorical(summary_df["noise_level"], categories=noise_order, ordered=True)
    summary_df = summary_df.sort_values(["filter", "noise_level"])
    summary_df.to_csv(OUT_DIR / "results_summary.csv", index=False)

    # Summary line charts

    # One chart per metric showing all filters across noise levels
    for metric in ["MSE_peak_height", "MSE_fwhm", "MSE_peak_count", "F1", "MSE_signal", "SNR_dB", "time_ms"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for filter_name in FILTERS:
            sub = summary_df[summary_df["filter"] == filter_name].set_index("noise_level")
            sub = sub.loc[noise_order]
            ax.plot(noise_order, sub[metric], marker="o",
                    color=FILTER_COLORS[filter_name], label=filter_name, linewidth=1.5)
        ax.set_xlabel("Noise level")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"summary_{metric}.png", dpi=150)
        plt.close()

    print(f"[info] Saved summary charts to {OUT_DIR}")

    # Confusion matrices

    # One row of confusion matrices per noise level, one matrix per filter
    filter_names = list(FILTERS.keys())
    for noise_label in noise_order:
        fig, axes = plt.subplots(1, len(filter_names), figsize=(3.5 * len(filter_names), 3.5))
        for ax, filter_name in zip(axes, filter_names):
            c  = cm_counts[(filter_name, noise_label)]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            # TN is undefined at peak level because there are infinite non-peak bins
            tn     = 0
            matrix = np.array([[tp, fn], [fp, tn]])
            im     = ax.imshow(matrix, cmap="Blues")
            # Write the count number inside each cell of the matrix
            for r in range(2):
                for col in range(2):
                    ax.text(col, r, matrix[r, col], ha="center", va="center", fontsize=12,
                            color="white" if matrix[r, col] > matrix.max() / 2 else "black")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred Pos", "Pred Neg"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Actual Pos", "Actual Neg"])
            ax.set_title(filter_name, color=FILTER_COLORS[filter_name], fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Confusion matrices — {noise_label}", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_{noise_label}.png", dpi=150)
        plt.close()

    print(f"[info] Saved confusion matrices to {OUT_DIR}")

    # Final summary

    # Print the averaged metrics table to the terminal
    print(f"\n{summary_df.to_string(index=False)}")

    # Report total wall-clock time for the whole benchmark run
    total_s = time.perf_counter() - benchmark_start
    print(f"\n[info] Total benchmark time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
