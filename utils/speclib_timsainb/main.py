
import __init__
from __init__ import *
from pretty_spectrogram import pretty_spectrogram
from invert_pretty_spectrogram import invert_pretty_spectrogram
from butter import butter_bandpass_filter

import os

# 0. load wav
DATAROOT = "/Users/liaoyuanda/Desktop/DAE_Cocktail/wav_clips"
WAVNAME = "sa1f.wav"
Fs, x_raw = wavfile.read(os.path.join(DATAROOT, WAVNAME))
# x = butter_bandpass_filter(x_raw, const['lowcut'], const['highcut'], Fs, order=1)

print('length = {}'.format(np.shape(x_raw)[0] / Fs))

# 1. plot specgram
wav_spectrogram = pretty_spectrogram(x_raw.astype('float64'), fft_size = const['fft_size'], \
                                   step_size = const['step_size'], log = True, thresh = const['spec_thresh'])
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
cax = ax.matshow(np.transpose(wav_spectrogram), \
         interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
fig.colorbar(cax)
plt.title('Original Spectrogram')
# plt.imsave(os.path.join(DATAROOT, "original"), np.transpose(wav_spectrogram[:,::-1]))
plt.show()

# 2. Invert from the spectrogram back to a waveform
recovered_audio_orig = invert_pretty_spectrogram(wav_spectrogram, fft_size = const['fft_size'],
                                            step_size = const['step_size'], log = True, n_iter = 10)
wavfile.write(os.path.join(DATAROOT, "sa1f_recover.wav"), Fs, recovered_audio_orig)

# 3. Make a spectrogram of the inverted audio (for visualization)
inverted_spectrogram = pretty_spectrogram(recovered_audio_orig.astype('float64'), fft_size = const['fft_size'], 
                                   step_size = const['step_size'], log = True, thresh = const['spec_thresh'])
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(18,4))
cax = ax.matshow(np.transpose(inverted_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
fig.colorbar(cax)
plt.title('Recovered Spectrogram')
# plt.imsave(os.path.join(DATAROOT, "recovered.png"), np.transpose(inverted_spectrogram[:,::-1]))
plt.show()
