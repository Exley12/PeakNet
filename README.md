# Peaknet
Convert your noisey spectra into a nice clean spectra today!
Just import peaknet then use clean_spectra = peaknet(spectra) and you will have a nice clean spectra.

Note the spectra must be a 1D array of length L where L is the length and 10 < L < 4096.
It could be a list, numpy array, tuple, tensor...

Example
```python
import peaknet
spectrum = [0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.3, 0.15, 0.2, 0.1, 0.05, 0.1, 0.15, 0.1]
predictions = peaknet(spectrum)  # [0.0, 0.0, 0.0, 0.15, 0.7, 0.9, 0.75, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
