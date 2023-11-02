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


dir = sorted(glob.glob("/workspace/data/dataset/slakh2100_flac_redux/test/*/mix_16k.wav"))
handler = InferenceHandler(
    root_path='./pretrained',
    # weight_path='logs/results_norm_20230823_113404/version_0/checkpoints/epoch=406-step=262515-val_loss=1.1770.pth',
    # weight_path='logs/results_norm_20230824_224103/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_randomorder_20230828_191936/version_0/checkpoints/last.pth',   # BEST
    weight_path='./pretrained/mt3.pth',
    # weight_path='logs/results_norm_randomorder_pix2seq_20230908_005826/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_randomorder_xl_20230909_232144/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_randomorder_xl_test_20230911_224525/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_randomorder_xl_test_20230912_100557/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_xl2_20230914_012521/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_xl2_inst_2048_20230921_214535/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_xl2_512_20230915_225945/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_xl2_inst_20230919_123531/version_0/checkpoints/last.pth',
    device=torch.device('cuda')
)

def func(fname):
    audio, _ = librosa.load(fname, sr=16000)
    return audio

pbar = tqdm(total=len(dir))
print("Total songs:", len(dir))

exp_tag_name = "slakh_baseline"
for fname in tqdm(dir):
    audio = func(fname)
    name = fname.split("/")[-2]
    handler.inference(
        audio, 
        fname, 
        outpath=os.path.join(exp_tag_name, 
                            name, 
                            "mix.mid"
                            ),
        batch_size=8        # changing this might affect results of sequential-inference models (e.g. XL).
    )