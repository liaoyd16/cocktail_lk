
import Meta
from speclib_timsainb.pretty_spectrogram import pretty_spectrogram
from speclib_timsainb.invert_pretty_spectrogram import invert_pretty_spectrogram

import __init__
from __init__ import *

USE_SPECGRAM = "speclib"

def spectrogram(slic):
    if USE_SPECGRAM == ["speclib", "pyplot"][0]:
        ans = pretty_spectrogram(slic.astype('float64'),\
                fft_size=2048,\
                step_size=128,\
                log = True,\
                thresh = Meta.data_meta['spec_thresh'])
        ans = ans.transpose()[:682][::-1] # see local: Desktop/speclib_timsainb/main1.py
        xs = np.linspace(0, 128, num=ans.shape[1])
        #ys = np.logspace(0, 8, num=ans.shape[0], base=2) - 1
        ys = np.linspace(0, 255, num=682)
        ans = scipy.interpolate.interp2d(x=xs, y=ys, z=ans)(np.arange(128), np.arange(256))

        return ans

    else:
        spec, _, _, _= plt.specgram(slic, Fs=Meta.data_meta['Fs'], NFFT=2048, noverlap=1900)
        plt.close('all')
        gc.collect()

        spec = cv2.resize(spec[:600, :][::-1], (128, 256), interpolation=cv2.INTER_CUBIC)
        spec = spec / np.mean(spec)
        spec[spec > 1] = 1
        spec = Meta.mel(spec / np.mean(spec))

        return spec
