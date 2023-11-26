# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio spectrogram functions."""

import dataclasses
import torch
from torchaudio.transforms import MelSpectrogram
import librosa
import numpy as np

# this is to suppress a warning from torch melspectrogram
import warnings
warnings.filterwarnings("ignore")

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames using librosa."""
    if samples.shape[0] % spectrogram_config.hop_width != 0:
        samples = np.pad(
            samples, 
            (0, spectrogram_config.hop_width - samples.shape[0] % spectrogram_config.hop_width), 
            'constant',
            constant_values=0
        )
    return librosa.util.frame(
        samples,
        frame_length=spectrogram_config.hop_width,
        hop_length=spectrogram_config.hop_width,
        axis=-1).T


def compute_spectrogram(
    samples, 
    spectrogram_config,
    use_tf_spectral_ops=False,
):
    """
    Compute a mel spectrogram.
    Due to multiprocessing issues running TF and PyTorch together, we use librosa
    and only keep `spectral_ops.compute_logmel` for evaluation purposes.
    """
    if use_tf_spectral_ops:
        # NOTE: we only keep this for evaluating existing models
        # This is because I find even with an equivalent PyTorch / librosa implementation 
        # that gives close-enough results (melspec MAE ~ 2e-3), the model output is still affected badly.
        # lazy load
        from ddsp import spectral_ops
        overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
        return spectral_ops.compute_logmel(
            samples,
            bins=spectrogram_config.num_mel_bins,
            lo_hz=MEL_LO_HZ,
            overlap=overlap,
            fft_size=FFT_SIZE,
            sample_rate=spectrogram_config.sample_rate)
    else:
        transform = MelSpectrogram(
            sample_rate=spectrogram_config.sample_rate,
            n_fft=FFT_SIZE,
            hop_length=spectrogram_config.hop_width,
            n_mels=spectrogram_config.num_mel_bins,
            f_min=MEL_LO_HZ,
            power=1.0,
        )
        samples = torch.from_numpy(samples).float()
        S = transform(samples)
        S[S<0] = 0
        S = torch.log(S + 1e-6)
        return S.numpy().T


def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""
    return np.reshape(frames, (-1,))


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins
