"""
Create train/validation/test split for ComMU dataset with equal distribution of instruments.
"""
import glob
import os
import pandas as pd
from collections import defaultdict
from const import get_inst_dict


os.makedirs("/data/datasets/ComMU/dataset_processed", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_audio_v2", exist_ok=True)     # for rendering later
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_midi_v2", exist_ok=True)


inst_dict = get_inst_dict()

# group original dataset by instrument
df = pd.read_csv('/data/datasets/ComMU/dataset/commu_meta.csv')
inst_to_idname_dict = defaultdict(list)
for index, row in df.iterrows():
    instrument, idname = row["inst"], row["id"]
    for key in inst_dict:
        if key in instrument:
            inst_to_idname_dict[key].append(idname)

for key in inst_to_idname_dict:
    inst_to_idname_dict[key] = sorted(inst_to_idname_dict[key])

sorted_keys = sorted(inst_to_idname_dict, key=lambda x: len(inst_to_idname_dict[x]), reverse=True)
splits = {}
for key in sorted_keys:
    inst_num = len(inst_to_idname_dict[key])
    splits[key] = {
        "train": inst_to_idname_dict[key][:int(inst_num * 0.9)],
        "val": inst_to_idname_dict[key][int(inst_num * 0.9):int(inst_num * 0.95)],
        "test": inst_to_idname_dict[key][int(inst_num * 0.95):]
    }

for key in splits:
    print(key, len(splits[key]["train"]), len(splits[key]["val"]), len(splits[key]["test"]))
    print(key, splits[key]["train"][:5], splits[key]["val"][:5], splits[key]["test"][:5])
    for idname in splits[key]["train"]:
        df.loc[df.id == idname, 'split_data'] = "train"
    for idname in splits[key]["val"]:
        df.loc[df.id == idname, 'split_data'] = "val"
    for idname in splits[key]["test"]:
        df.loc[df.id == idname, 'split_data'] = "test"
    

df.to_csv("/data/datasets/ComMU/dataset_processed/commu_meta_v2.csv", index=False)
df = pd.read_csv("/data/datasets/ComMU/dataset_processed/commu_meta_v2.csv")
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_audio_v2/train/", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_audio_v2/val/", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_midi_v2/train/", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_midi_v2/val/", exist_ok=True)
os.makedirs("/data/datasets/ComMU/dataset_processed/commu_midi_v2/test/", exist_ok=True)

for index, row in df.iterrows():
    idname = row["id"]
    split = row["split_data"]
    midi_path = glob.glob(f"/data/datasets/ComMU/dataset/commu_midi/*/raw/{idname}.mid")[0]
    if split == "train":
        os.rename(midi_path, "/data/datasets/ComMU/dataset_processed/commu_midi_v2/train/" + idname + ".mid")
    elif split == "val":
        os.rename(midi_path, "/data/datasets/ComMU/dataset_processed/commu_midi_v2/val/" + idname + ".mid")
    elif split == "test":
        os.rename(midi_path, "/data/datasets/ComMU/dataset_processed/commu_midi_v2/test/" + idname + ".mid")