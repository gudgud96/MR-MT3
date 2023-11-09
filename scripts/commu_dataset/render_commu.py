import pretty_midi
import soundfile as sf
import pandas as pd
"""
Render audio for ComMU dataset using FluidSynth.
"""
from tqdm import tqdm
import os
import json
import librosa
from const import get_inst_dict


inst_dict = get_inst_dict()

df = pd.read_csv('/data/datasets/ComMU/dataset_processed/commu_meta_v2.csv')
for index, row in tqdm(df.iterrows(), total=len(df)):
    path = f"/data/datasets/ComMU/dataset_processed/commu_midi_v2/{row['split_data']}/{row['id']}.mid"
    
    # from Kin Wai's `note2wav`, but (i) removed the note velocity part; (ii) changed program number
    midi_data = pretty_midi.PrettyMIDI(path)

    expected_program_number = -1
    for key in inst_dict:
        if key in row['inst']:
            expected_program_number = inst_dict[key]
            break
    
    if expected_program_number == -1:
        print("problem", row["id"])
        continue
    
    # assert abs(midi_data.instruments[0].program - expected_program_number) == 1 or abs(midi_data.instruments[0].program - expected_program_number) == 82, f"{row['id']} {midi_data.instruments[0].program} {expected_program_number}"

    midi_data.instruments[0].program = expected_program_number
    midi_data.write(path)       # overwrite the midi file

    wav = midi_data.fluidsynth(fs=44100)
    wav = librosa.resample(wav, orig_sr=44100, target_sr=16000)     # let's try, because 16k quality is quite bad

    audio_folder = path.replace(f"{row['id']}.mid", "").replace("commu_midi_v2", "commu_audio_v2")
    sf.write(os.path.join(audio_folder, f"{row['id']}_16k.wav"), wav, 16000, "PCM_24")
