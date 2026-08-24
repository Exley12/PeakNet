# Author: Matthew Exley

# Imports
import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

# Project root (two levels up from this file), needed to import from peaknet/
# regardless of where this script gets run from
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Default spectrum length bounds, pulled from the shared config
from peaknet.config import max_spectrum_length
from peaknet.config import min_spectrum_length

def _make_peak_spectra_batch(count, L, n_peaks, min_width, max_width, min_height, max_height, noise_min, noise_max):
    """Generate `count` spectra all of length `L` in one vectorised pass."""
    # Bin index axis (0, 1, 2, ... L-1), shared across every spectrum in this batch
    x = np.arange(L, dtype=np.float32)
    # Random number of peaks (0 to n_peaks) drawn for each spectrum in the batch
    ks = np.random.randint(0, n_peaks + 1, size=count)
    # Most peaks any single spectrum has drawn this round, used to preallocate arrays for the whole batch
    max_k = int(ks.max()) if count > 0 else 0

    if max_k == 0:
        # No peaks drawn at all this round, so every spectrum is just flat zero
        masks = np.zeros((count, L), dtype=np.float32)
    else:

        # Everything below runs for ALL spectra x ALL possible peak slots at once,
        # shape (count, max_k), instead of looping - much faster in numpy
        # 20% chance of a "tight" width cap (L/8), otherwise a looser one (L/20)
        width_divisors = np.where(np.random.random((count, max_k)) < 0.2, 8, 20)

        # Width cap never allowed to collapse below 1 bin
        max_allowed = np.maximum(1.0, L / width_divisors)

        # Width ceiling for this peak: whichever is smaller, the config max or the length-based cap
        actual_max = np.minimum(max_width, max_allowed)

        # Floor never allowed to end up above the ceiling just picked
        actual_min = np.minimum(min_width, actual_max)

        # Width picked randomly somewhere between the floor and ceiling
        widths = actual_min + np.random.random((count, max_k)) * (actual_max - actual_min)

        # 20% chance to override with a deliberately narrow/sharp peak (width 1-3 bins)
        narrow = np.random.random((count, max_k)) < 0.2
        widths = np.where(narrow, np.random.uniform(1, 3, (count, max_k)), widths)

        # Peak centres kept away from the edges - margin scales with how wide the peak is
        edge_margins = np.minimum(3 * widths, (L - 1) / 2)

        # Room left in the middle for the centre to actually move around in
        centre_range = np.maximum(0.0, (L - 1) - 2 * edge_margins)

        # Centre placed randomly somewhere inside that safe middle region
        centres = edge_margins + np.random.random((count, max_k)) * centre_range

        # Random peak heights for every slot
        heights = np.random.uniform(min_height, max_height, (count, max_k))

        # Some spectra only "wanted" fewer peaks than max_k (that's what ks was for) -
        # so any slot beyond that spectrum's own peak count gets masked out to zero height
        active = np.arange(max_k)[None, :] < ks[:, None]
        heights = np.where(active, heights, 0.0)

        # Every Gaussian peak for every spectrum built at once: broadcast to (count, max_k, L),
        # then summed across the peak axis so overlapping peaks stack into one spectrum per row -> (count, L)
        peaks = heights[:, :, None] * np.exp(
            -0.5 * ((x[None, None, :] - centres[:, :, None]) / widths[:, :, None]) ** 2
        )
        masks = peaks.sum(axis=1).astype(np.float32)

    # The "mask" (clean ground truth) and "spectra" (noisy input) start out identical...
    spectra = masks.copy()
    # ...then get normalised so nothing exceeds 1.0 - tallest point in each spectrum found here
    max_signals = masks.max(axis=1, keepdims=True)
    # Only spectra that actually went above 1.0 get scaled down, the rest left alone
    scale = np.where(max_signals > 1.0, max_signals, 1.0)
    spectra /= scale
    masks /= scale

    # Random noise level picked per spectrum, then actual gaussian noise generated at that level
    noise_levels = np.random.uniform(noise_min * 0.2, noise_max * 0.2, size=(count, 1))
    noise = (np.random.normal(0, 1, (count, L)) * noise_levels).astype(np.float32)
    # Noise added to the input spectra only (masks stay clean as the training target)
    spectra = np.clip(spectra + noise, -0.2, 1).astype(np.float32)
    masks = np.clip(masks, 0, 1).astype(np.float32)
    return spectra, masks

def make_peak_dataset(outdir, L_min, L_max, N, n_peaks, min_width, max_width,
                      min_height, max_height, noise_min, noise_max):
    
    # Output folder creation
    os.makedirs(outdir, exist_ok=True)
    
    # Starting a timer 
    t0 = time.perf_counter()

    # Every spectrum gets a random length in [L_min, L_max] - this is what makes the dataset ragged
    lengths = np.random.randint(L_min, L_max + 1, size=N)

    # Output lists pre-allocated, filled in below in whatever order the groups come out
    spectra = [None] * N
    masks = [None] * N

    """
    inverse is a handy tool to use here
    lengths = np.random.randint(50, 60, size=5)
    print(lengths)
    [54, 51, 58, 55, 54] 
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)
    print(unique_lengths)
    [51, 54, 55, 58]
    print(inverse)
    [1, 0, 3, 2, 1]
    The inverse tells you the position of each original length in the unique_lengths
    """
    # Spectra grouped by length so same length groups can be generated in one fast vectorised batch,
    # instead of calling the generator N separate times - `inverse` maps each spectrum back to its group
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)

    for L_idx, L in tqdm(enumerate(unique_lengths), total=len(unique_lengths), desc="generating"):

        # Which spectra (by original index) belong to this length group
        idx = np.where(inverse == L_idx)[0]

        # Generate the whole group of same length spectra in one go
        batch_s, batch_m = _make_peak_spectra_batch(
            len(idx), int(L), n_peaks, min_width, max_width,
            min_height, max_height, noise_min, noise_max
        )

        # Scatter the batch results back into their original positions
        for j, i in enumerate(idx):
            spectra[i] = batch_s[j]
            masks[i] = batch_m[j]

    # End the timer        
    t1 = time.perf_counter()
    print(f"[info] generated {N} spectra in {t1 - t0:.2f}s ({N / (t1 - t0):.0f} spectra/s)")

    # dtype=object since the spectra are different lengths (ragged) - numpy can't store that as
    # a normal rectangular array, so loading these back in needs allow_pickle=True
    np.save(os.path.join(outdir, "peaks_spectra.npy"), np.array(spectra, dtype=object))
    np.save(os.path.join(outdir, "peaks_masks.npy"), np.array(masks, dtype=object))

    # Extra timer to show save time
    t2 = time.perf_counter()
    print(f"[info] saved {N} peak spectra to {outdir} in {t2 - t1:.2f}s")

if __name__ == "__main__":
    # Make the argument parser
    parser = argparse.ArgumentParser(description="Generate simple Peaknet peak spectra")
    # Set the output folder
    parser.add_argument("--outdir", default="./Data/dataset_small")
    # Set the spectrum min length
    parser.add_argument("--L_min", type=int, default=min_spectrum_length)
    # Set the spectrum max length
    parser.add_argument("--L_max", type=int, default=max_spectrum_length)
    # Set the number of spectra
    parser.add_argument("--N", type=int, default=300)
    # Set the max number of peaks per spectrum
    parser.add_argument("--n_peaks", "--n-peaks", type=int, default=10)
    # Set the minimum peak width in bins
    parser.add_argument("--min_width", "--min-width", type=float, default=1)
    # Set the maximum peak width in bins
    parser.add_argument("--max_width", "--max-width", type=float, default=30)
    # Set the minimum peak height
    parser.add_argument("--min_height", "--min-height", type=float, default=0.08)
    # Set the maximum peak height
    parser.add_argument("--max_height", "--max-height", type=float, default=0.85)
    # Set the minimum noise level
    parser.add_argument("--noise_min", "--noise-min", type=float, default=0.1)
    # Set the maximum noise level
    parser.add_argument("--noise_max", "--noise-max", type=float, default=1.0)
    # Read the arguments
    args = parser.parse_args()
    # Generation kicked off with whatever args ended up set (defaults or CLI overrides)
    make_peak_dataset(args.outdir, args.L_min, args.L_max, args.N, args.n_peaks, args.min_width, args.max_width, args.min_height, args.max_height, args.noise_min, args.noise_max)
