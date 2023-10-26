import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from inference import InferenceHandler
import torch
import glob
import os
from tqdm import tqdm
import librosa
import concurrent.futures
import traceback

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# dir = sorted(glob.glob("/data/slakh2100_flac_redux/test/*/mix_16k.wav"))

# ====== version 1 ====== #
# dir = sorted(glob.glob("/data/datasets/ComMU/dataset/commu_audio_v2/test/*.wav"))

# ====== version 2 ====== #
dir = sorted(glob.glob("/data/datasets/ComMU/dataset_processed/commu_audio_v2/val/*.wav"))

handler = InferenceHandler(
    root_path='./pretrained',
    # weight_path='logs/results_commu_20231017_210115/version_0/checkpoints/epoch=94-step=456855-val_loss=0.6072.pth',
    # weight_path='logs/results_commu_20231020_154040/version_0/checkpoints/epoch=30-step=38812-val_loss=0.6639.pth',
    # weight_path='logs/results_commu_20231020_154040/version_0/checkpoints/last.pth',
    weight_path='logs/results_commu_20231023_181007/version_0/checkpoints/last.pth',
    # weight_path='logs/results_commu_20231023_181007/version_0/checkpoints/epoch=35-step=45072-val_loss=0.6400.pth',
    # weight_path='pretrained/mt3.pth',
    device=torch.device('cuda')
)

def func(fname):
    audio, _ = librosa.load(fname, sr=16000)
    return audio

pbar = tqdm(total=len(dir))
print("Total songs:", len(dir))

exp_tag_name = "commu_v4_valset"
for fname in tqdm(dir):
    audio = func(fname)
    name = fname.split("/")[-1]
    handler.inference(
        audio, 
        fname, 
        outpath=os.path.join(exp_tag_name, 
                            name.replace(".wav", ".mid")
                            ),
        batch_size=8,        # changing this might affect results of sequential-inference models (e.g. XL).
        max_length=256
    )