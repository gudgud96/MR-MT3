import glob
import miditoolkit
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

lst = [
        "/data/slakh2100_flac_redux/test/",
        "../../midis/slakh_baseline/",
        "../../outputs/exp_baseline_librosa_ep800/",
        "../../outputs/exp_segmemV2_prev_context=0/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=32/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64_prevaug_frame=3/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64_prevaug_frame=8/slakh_mt3_official/",
    ]

plt.figure(figsize=(20, 5))
res_dict = {}
for k in lst:
    print(k)
    res_dict[k] = {}
    if k == "/data/slakh2100_flac_redux/test/":
        midis = sorted(glob.glob(k + "*/all_src_v2.mid"))
    else:
        midis = sorted(glob.glob(k + "*/*.mid"))
    for midi in tqdm(midis):
        midi_id = midi.split("/")[-2]
        midi_obj = miditoolkit.MidiFile(midi)
        res_dict[k][midi_id] = len(set([k.program for k in midi_obj.instruments]))

ids = sorted(list(res_dict[lst[0]].keys()))
for l in lst:
    lengths = []
    for k in ids:
        lengths.append(res_dict[l][k])
    print(l, np.mean(lengths), "+/-", np.std(lengths))

