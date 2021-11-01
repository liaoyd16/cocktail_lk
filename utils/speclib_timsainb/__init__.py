
# iPython specific stuff
# matplotlib inline
# Packages we're using
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import numpy as np

### Parameters ###
const = {
'fft_size': 2048 ,# window size for the FF
'step_size': 128, # distance to slide along the window (in time
'spec_thresh': 4 ,# threshold for spectrograms (lower filters out more noise
'lowcut': 500 ,# Hz # Low cut for our butter bandpass filte
'highcut': 15000, # Hz # High cut for our butter bandpass filte
}

'''
butter (0.2)
-----------------------------
butter_bandpass (0.1)
butter_bandpass_filter (0.0)
overlap (1.2)
stft (1.1)
pretty_spectrogram (1.0)
invert_pretty_spectrogram (2.0)
iterate_invert_spectrogram (2.1)
invert_spectrogram (2.2)
xcorr_offset (2.3)
make_mel (3.x)
mel_to_spectrogram (3.x)
_hz2mel (3.x)
_mel2hz (3.x)
get_filterbanks (3.x)
create_mel_filter (3.0)
-----------------------------
raw_samples
 |
 v : butter_bandpass_filter
samples
 ^ : invert_pretty_spectrogram
 |
 v : pretty_spectrogram
spectrogram
'''