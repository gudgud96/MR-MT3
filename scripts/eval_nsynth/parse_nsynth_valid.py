import json
import glob
import miditoolkit
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tag_name", type=str)
parser.add_argument("--path", type=str)
args = parser.parse_args()

with open("/data/nsynth-valid/examples.json") as f:
    instruments = json.load(f)

instrument_source = ["acoustic", "electronic", "synthetic"]
# midi_class, program_num // 8
instrument_family = {
    "bass": 4,
    "brass": 7,
    "flute": 9,
    "guitar": 3,
    "keyboard": 0,
    "mallet": 1,
    "organ": 2,
    "reed": 8,
    "string": 5,
    "synth_lead": 10,
    "vocal": 0          # vocal shall be treated separately, let's inspect what the program number is
}

midi_class = [
    "piano", 
    "chromatic_percussion", 
    "organ", 
    "guitar", 
    "bass", 
    "strings", 
    "ensemble", 
    "brass", 
    "reed",
    "pipe",
    "synth_lead",
    "synth_pad",
    "synth_fx",
    "ethnic",
    "percussive",
    "sound_fx"
]

for key in sorted(list(instrument_family.keys())):
    print(f"nsynth family: {key} --> midi_class family: {midi_class[instrument_family[key]]}")


# 1. how many instrument tracks per transcription
# 2. accuracy of pitch
# 3. instrument F1 score

result_dict = {}
dir = sorted(glob.glob(args.path))

for item in tqdm(dir):
    name = item.split("/")[-1].replace(".mid", "")
    
    result_dict[name] = {}

    dic = instruments[name] 
    family_str, expected_pitch = dic["instrument_family_str"], dic["pitch"]
    expected_program_family = instrument_family[family_str]
    result_dict[name]["expected_instrument"] = midi_class[expected_program_family] if family_str != "vocal" else "vocal"
    result_dict[name]["expected_pitch"] = expected_pitch

    # print(f"{name}, family: {family_str}, pitch: {expected_pitch}, midi_class: {midi_class[expected_program_family]}")

    if not os.path.exists(item):
        # no midi annotation, this means no instrument & note are detected.
        result_dict[name]["predicted"] = {}
        result_dict[name]["predicted"]["num_instruments"] = 0
        result_dict[name]["predicted"]["events"] = []
        continue

    else:
        mid_obj = miditoolkit.MidiFile(item)
        result_dict[name]["num_tracks"] = len([inst for inst in mid_obj.instruments if inst.is_drum == False])
        inst_note_dict = {}
        for inst in mid_obj.instruments:
            # we ignore drum tracks, and group tracks under the same midi class
            if not inst.is_drum:    
                pitches = [n.pitch for n in inst.notes]
                if inst.program  // 8 not in inst_note_dict:
                    inst_note_dict[inst.program // 8] = []
                inst_note_dict[inst.program // 8] += pitches
        
        result_dict[name]["predicted"] = {}
        result_dict[name]["predicted"]["num_instruments"] = len(inst_note_dict)
        result_dict[name]["predicted"]["events"] = []
        for predicted_program_num in inst_note_dict:
            res = {}
            res["instrument"] = midi_class[predicted_program_num]
            res["pitch"] = sorted(list(set(inst_note_dict[predicted_program_num])))
            result_dict[name]["predicted"]["events"].append(res)
        
        # print("predicted:")
        # for predicted_program_num in inst_note_dict:
        #     print(f"midi_class: {midi_class[predicted_program_num]}, pitch(es): {inst_note_dict[predicted_program_num]}")
    
with open(f"{args.tag_name}.json", "w+") as f:
    json.dump(result_dict, f, indent=2)