
import os
from scipy.io import wavfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def _get_audios(audio_dir):
    audio_list = os.listdir(audio_dir)
    ans = []
    for audio in audio_list:
        if ".wav" in audio: ans.append(audio)
    return ans

AUDIO_DIR = "/Users/liaoyuanda/Desktop/DAE_Cocktail/wav_clips/"
audios = _get_audios(AUDIO_DIR)

for audio in audios:
    Fs, x = wavfile.read(AUDIO_DIR + audio)
    specgram, freqs, t, _ = plt.specgram(x, Fs=Fs, NFFT=2048, noverlap=1900)
    plt.show()

    plt.imshow(specgram)
    plt.show()

    plt.hist(specgram.reshape(-1), bins = 100)
    plt.yscale("log")
    plt.show()