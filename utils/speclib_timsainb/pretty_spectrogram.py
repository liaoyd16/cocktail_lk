
import __init__
from __init__ import *
import sys
import os
import Meta
sys.path.append(os.path.join(Meta.PROJ_ROOT, "utils/speclib_timsainb"))
from butter import butter_bandpass_filter
from stft import stft

'''
butter_bandpass (0.1)
butter_bandpass_filter (0.0)
overlap (1.2)
_stft (1.1)
pretty_spectrogram (1.0)
'''

_EPS = 5e-2
def pretty_spectrogram(d,log = True, thresh= 5, fft_size = 512, step_size = 64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=True,
        compute_onesided=True))

    if log == True:
        #specgram /= (_EPS + np.mean(abs(specgram)))
        specgram = np.log10(specgram) # take log
        specgram[specgram < thresh] = thresh
        specgram -= thresh
        # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram
