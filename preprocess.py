import sys
from pathlib import Path
from scipy.fft import rfft
from scipy.signal.windows import blackmanharris
import soundfile as sf
import numpy as np


nr_window_sz = 128
wd_window_sz = 1024
nr_window = blackmanharris(nr_window_sz)
wd_window = blackmanharris(wd_window_sz)


def compute_fft(signal):
    if len(signal) < window_size:
        signal = np.pad(signal, (0, window_size - len(signal)), mode='constant')
    else:
        signal = signal[:window_size]

    windowed_nr = signal * window

    # Compute FFT
    fft_vals = rfft(windowed_signal)

    return fft_vals


def main(argv):
    path = Path('../data').resolve()
    spectr = Path('../spectr')
    spectr.mkdir(parents=True, exist_ok=True)
    spectr = spectr.resolve()
    nr_sfft = rfft(nr_w, nr_hop, 44100)
    wd_sfft = ShortTimeFFT(wd_w, wd_hop, 44100)
    for p in path.glob('**/*.flac'):
        print(p.parts[-1])
        print(p.parts[-2])
        data, _ = sf.read(p)
        nr_sx = nr_sfft.stft(data)
        wd_sx = wd_sfft.stft(data)
        print(nr_sx)
        break


if __name__ == "__main__":
    main(sys.argv)
