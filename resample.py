import librosa
import glob
import soundfile as sf
import os
import concurrent.futures
from tqdm import tqdm
import traceback
from shutil import copy2

def func(fname):
    audio, sr = librosa.load(fname, sr=None)
    if sr != 16000:
        print("resampling", fname)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(
        fname.replace(".flac", "_16k.wav"),
        audio,
        16000,
        "PCM_24"
    )

dir = sorted(glob.glob("/data/slakh2100_flac_redux/test/**/mix.flac"))
pbar = tqdm(total=len(dir))


with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    # Start the load operations and mark each future with its URL
    future_to_fname = {executor.submit(func, fname): fname for fname in dir}
    for future in concurrent.futures.as_completed(future_to_fname):
        try:
            fname = future_to_fname[future]
            audio = future.result()
            pbar.update()
        except Exception as e:
            traceback.print_exc()    