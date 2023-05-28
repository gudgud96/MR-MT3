from inference import InferenceHandler
import torch
import glob
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import json
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dir = sorted(glob.glob("/data/slakh2100_flac_redux/validation/*/mix.flac"))[:100]
dir2 = sorted(glob.glob("loops_data/*/mix.wav"))
dir3 = sorted(glob.glob("/data/audioset-processing/output/electronic music/*.flac"))


x = glob.glob("/data/fs_analysis/*.json")    
ids = []                                                                                                                                                                                                                                                                                                                       
for elem in tqdm(x):                                                                                                                                                           
    with open(elem) as f:                                                                                                                                                     
        j = json.load(f)   
    is_pass = True
    j["tags"] = [x.lower() for x in j["tags"]]
    for keyword in ["drum", "drums", "percussion", "beat", "beatloop"]:
        if keyword in j["tags"]:
            is_pass = False
            break
    if is_pass:
        print(j["tags"])
        ids.append(elem.split("/")[-1].replace(".json", ""))

dir4 = []
for idx in ids:
    dir4.extend(glob.glob(f"/data/audio/wav/{idx}_*.wav"))
    if len(dir4) > 100:
        break

handler = InferenceHandler('./pretrained', device=torch.device('cuda'))

def get_vec(dir, handler):
    vecs = []
    for fname in tqdm(dir):
        name = fname.split("/")[1]
        encoder_output = handler.get_encoder_outputs(fname,)
        encoder_output = encoder_output.reshape(-1, 512)
        encoder_output = torch.mean(encoder_output, dim=0)
        vecs.append(encoder_output)
    return torch.stack(vecs)


if os.path.exists("vecs1.npy"):
    vecs1 = np.load("vecs1.npy")
    vecs2 = np.load("vecs2.npy")
    vecs3 = np.load("vecs3.npy")
    vecs4 = np.load("vecs4.npy")

print(vecs1.shape, vecs2.shape, vecs3.shape, vecs4.shape)
# else:
# vecs1 = get_vec(dir, handler)
# vecs1 = vecs1.cpu().detach().numpy()
# np.save("vecs1.npy", vecs1)

# vecs2 = get_vec(dir2, handler)
# vecs2 = vecs2.cpu().detach().numpy()
# np.save("vecs2.npy", vecs2)

# vecs3 = get_vec(dir3, handler)
# vecs3 = vecs3.cpu().detach().numpy()
# np.save("vecs3.npy", vecs3)

# vecs4 = get_vec(dir4, handler)
# vecs4 = vecs4.cpu().detach().numpy()
# np.save("vecs4.npy", vecs4)

vecs = np.concatenate([vecs1, vecs2, vecs4], axis=0)
labels = np.concatenate([
    np.zeros(vecs1.shape[0],),
    np.ones(vecs2.shape[0],),
    # np.ones(vecs3.shape[0],) * 2,
    np.ones(vecs4.shape[0],) * 2,
], axis=0)
labels = labels.astype(np.int)

from sklearn.manifold import TSNE

for p in [5, 10, 20, 30, 40, 50, 60, 80, 100]:
    vecs_tsne = TSNE(n_components=2, perplexity=p).fit_transform(vecs)
    sns.scatterplot(
        x=vecs_tsne[:, 0],
        y=vecs_tsne[:, 1],
        hue=labels,
        palette=['blue','orange', 'red']
    )
    plt.savefig(f"embeddings_{p}.png")
    plt.close()