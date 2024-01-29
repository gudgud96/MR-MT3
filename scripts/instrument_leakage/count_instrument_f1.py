import glob
import miditoolkit
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

ground_truth_dir =  "/data/slakh2100_flac_redux/test/"
lst = [
        "/data/slakh2100_flac_redux/test/",
        "../../midis/slakh_baseline/",
        "../../outputs/exp_segmemV2_prev_context=0/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=32/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64_prevaug_frame=3/slakh_mt3_official/",
        "../../outputs/exp_segmemV2_prev_context=64_prevaug_frame=8/slakh_mt3_official/",
    ]

res_dict = {}
for k in lst:
    num_instruments = []
    precs = []
    recalls = []
    f1s = []

    res_dict[k] = {}
    if k == "/data/slakh2100_flac_redux/test/":
        midis = sorted(glob.glob(k + "*/all_src_v2.mid"))
    else:
        midis = sorted(glob.glob(k + "*/*.mid"))
    for midi in tqdm(midis):
        gt_midi = midi.replace(k, ground_truth_dir).replace("mix.mid", "all_src_v2.mid")
        midi_id = midi.split("/")[-2]

        midi_obj_pred = miditoolkit.MidiFile(midi)
        midi_obj_gt = miditoolkit.MidiFile(gt_midi)

        pred_instruments = sorted(list(set([k.program for k in midi_obj_pred.instruments])))
        gt_instruments = sorted(list(set([k.program for k in midi_obj_gt.instruments])))

        num_instruments.append(len(pred_instruments))

        # following: https://github.com/craffel/mir_eval/blob/main/mir_eval/transcription.py#L466
        matched = [k for k in pred_instruments if k in gt_instruments]
        precision = len(matched) / len(pred_instruments)
        recall = len(matched) / len(gt_instruments)
        f1 = 2 * precision * recall / (precision + recall)
        precs.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    print("{}: avg_num_insts: {:.4} +/- {:.4}, prec: {:.4} rec: {:.4} f1: {:.4}\n".format(
        k, np.mean(num_instruments), np.std(num_instruments), np.mean(precs), np.mean(recalls),
        np.mean(f1s)
    ))