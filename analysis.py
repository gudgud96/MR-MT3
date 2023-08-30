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

# dir = sorted(glob.glob("loops_data/*/mix.wav"))
# dir = sorted(glob.glob("/data/nsynth-valid/audio/organ_electronic_001-036-075.wav"))
# dir = sorted(glob.glob("/data/nsynth-valid/audio/*.wav"))
dir = sorted(glob.glob("/data/slakh2100_flac_redux/test/*/mix_16k.wav"))
handler = InferenceHandler(
    root_path='./pretrained',
    # weight_path='results/001/version_0/checkpoints/epoch=394-step=254775-val_loss=1.2673.pth',
    # weight_path='results_norm_20230803_231319/version_0/checkpoints/epoch=394-step=254775-val_loss=1.1790.pth',
    # weight_path='results_adv_20230805_114334/version_0/checkpoints/epoch=175-step=113520-val_loss=1.5884.pth',
    # weight_path='logs/results_adv_20230812_130025/version_0/checkpoints/epoch=392-step=253485-val_loss=1.1837.pth',
    # weight_path='logs/results_adv_embd_epsilon=0.001_advratio=0.5/version_0/checkpoints/epoch=392-step=253485-val_loss=1.1913.pth',
    # weight_path='logs/results_vat_epsilon=1e-5/version_0/checkpoints/epoch=311-step=201240-val_loss=1.1866.pth',
    # weight_path='logs/results_vat_epsilon=1e-5/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_20230823_113404/version_0/checkpoints/epoch=406-step=262515-val_loss=1.1770.pth',
    # weight_path='logs/results_norm_20230824_224103/version_0/checkpoints/last.pth',
    # weight_path='logs/results_norm_redtoken_20230827_152251/version_0/checkpoints/last.pth',
    weight_path='logs/results_norm_randomorder_20230828_191936/version_0/checkpoints/last.pth',
    # weight_path='./pretrained/mt3.pth',
    device=torch.device('cuda')
)

def func(fname):
    audio, _ = librosa.load(fname, sr=16000)
    return audio

pbar = tqdm(total=len(dir))
print("Total songs:", len(dir))
for fname in tqdm(dir):
    audio = func(fname)
    name = fname.split("/")[-2]
    handler.inference(
        audio, 
        fname, 
        outpath=os.path.join("slakh_recon_randomorder", 
                            name, 
                            "mix.mid"
                            ),
        batch_size=16
    )

# with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#     # Start the load operations and mark each future with its URL
#     future_to_fname = {executor.submit(func, fname): fname for fname in dir}
#     for future in concurrent.futures.as_completed(future_to_fname):
#         try:
#             fname = future_to_fname[future]
#             name = fname.split("/")[-2]
#             audio = future.result()
#             handler.inference(
#                 audio, 
#                 fname, 
#                 outpath=os.path.join("slakh_test_v1", 
#                                     name, 
#                                     "mix.mid"
#                                     ),
#                 batch_size=16
#             )
#             pbar.update()
#         except Exception as e:
#             traceback.print_exc()