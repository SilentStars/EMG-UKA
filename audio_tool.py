import soundfile as sf
import librosa
import torch
import numpy as np


mel_basis = {}
hann_window = {}

def normalize_volume(audio):
    rms = librosa.feature.rms(audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0: # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def load_audio(filename, start=None, end=None, max_frames=None, renormalize_volume=False):
    audio, r = sf.read(filename)

    if len(audio.shape) > 1:
        audio = audio[:,0] # select first channel of stero audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if renormalize_volume:
        audio = normalize_volume(audio)
    if r == 16000:
        audio = librosa.resample(y=audio, orig_sr=16000, target_sr=22050)
    else:
        assert r == 22050
    audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
    pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
    mspec = pytorch_mspec.squeeze(0).T.numpy()
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames,:]
    return mspec


filename = 'EMG-UKA-Full-Corpus/audio/551/001/a_551_0001.wav'
mspec = load_audio(filename)
def normalize(x):
    x = np.asarray(x)
    arr = (x - x.min()) / (x.max() - x.min())
    return np.subtract(np.multiply(arr, 2), 1)
mspec = normalize(mspec)
print(mspec)