import __init__
from __init__ import *
from Meta import model_meta, device
import librosa

class SpeechEmbedder(nn.Module):
    
    def __init__(self):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(model_meta['embedder']['nmels'], model_meta['embedder']['hidden'], \
                                  num_layers=model_meta['embedder']['num_layer'], batch_first=True)
        self.LSTM_stack.to(device)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(model_meta['embedder']['hidden'], model_meta['embedder']['proj'])
        self.projection.to(device)

        self.window_length = int(model_meta['embedder']['window'] * model_meta['embedder']['sr'])
        self.hop_length = int(model_meta['embedder']['hop'] * model_meta['embedder']['sr'])
        self.duration = model_meta['embedder']['tisv_frame'] \
                 * model_meta['embedder']['hop']\
                 + model_meta['embedder']['window']

    def mfccs_and_spec(self, sound, wav_process=True):
        # Cut silence and fix length
        if wav_process == True:
            sound, index = librosa.effects.trim(np.array(sound, dtype=float), frame_length=self.window_length, hop_length=self.hop_length)
            length = int(model_meta['embedder']['sr'] * self.duration)
            sound = librosa.util.fix_length(sound, length)

        spec = librosa.stft(sound, n_fft=model_meta['embedder']['nfft'], hop_length=self.hop_length, win_length=self.window_length)
        mag_spec = np.abs(spec)

        mel_basis = librosa.filters.mel(model_meta['embedder']['sr'], model_meta['embedder']['nfft'], n_mels=model_meta['embedder']['nmels'])
        mel_spec = np.dot(mel_basis, mag_spec)

        mel_db = librosa.amplitude_to_db(mel_spec).T

        return mel_db
        # mel_db: of size (data_points, n_mel=1+nfft/2)

    def embed(self, slices):   # (batch, len)
        # input
        xs = []
        for slic in slices:
            xs.append(self.mfccs_and_spec(slic))  #(batch, frames, n_mels)
        if torch.cuda.is_available(): xs = torch.cuda.FloatTensor(xs)
        else: xs = torch.Tensor(xs)

        embeddings = self.forward(xs)
        return embeddings

    def forward(self, x):      #(batch, frames, hidden)
        x, _ = self.LSTM_stack(x.float())
        # only use last frame
        x = x[:,x.size(1)-1]              #(batch, hidden)
        x = self.projection(x.float())    #(batch, proj)
        x = x / torch.norm(x, dim=1).unsqueeze(1) #(batch, proj)
        return x