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
# dir = sorted(glob.glob("babyslakh_16k/Track00002/mix.wav"))
dir = sorted(glob.glob("/data/slakh2100_flac_redux/test/*/mix_16k.wav"))
handler = InferenceHandler('./pretrained2', device=torch.device('cuda'))

# for fname in tqdm(dir):
#     name = fname.split("/")[-2]
#     name = "Track01876"
#     print(name)
#     audio, _ = librosa.load("Track01876.wav", sr=16000, mono=True)
#     handler.inference(audio, "Track01876.wav", outpath=os.path.join("out", name, "mix.mid"))
#     break

def func(fname):
    name = fname.split("/")[1]
    audio, _ = librosa.load(fname, sr=16000)
    return audio

pbar = tqdm(total=len(dir))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Start the load operations and mark each future with its URL
    future_to_fname = {executor.submit(func, fname): fname for fname in dir}
    for future in concurrent.futures.as_completed(future_to_fname):
        try:
            fname = future_to_fname[future]
            name = fname.split("/")[-2]
            audio = future.result()
            handler.inference(
                audio, 
                fname, 
                outpath=os.path.join("slakh2100_flac_redux_test_16k_retrain", 
                                    name, 
                                    "mix.mid"
                                    ),
                batch_size=64
            )
            pbar.update()
        except Exception as e:
            traceback.print_exc()