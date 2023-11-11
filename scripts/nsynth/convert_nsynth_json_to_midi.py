import json
import os
import pretty_midi
from tqdm import tqdm

# change the following path accordingly
prefix_path = "/data/nsynth-valid/"
with open(os.path.join(prefix_path, "examples.json")) as f:
    instruments = json.load(f)

instrument_mapping = {
    "bass": 32,             # bass
    "brass": 56,            # brass
    "flute": 72,            # pipe
    "guitar": 24,           # guitar
    "keyboard": 0,          # piano
    "mallet": 8,            # chromatic percussion
    "organ": 16,            # organ
    "reed": 64,             # reed
    "string": 40,           # strings
    "synth_lead": 80,       # synth lead
}

# here, we map `instrument_family_str` in the json file to the midi class number in the MIDI standard
# we use the first midi class number within the instrument family, e.g. "keyboard" --> 0
os.makedirs(os.path.join(prefix_path, "midi"), exist_ok=True)
for key in tqdm(instruments):

    # skip vocals, as other datasets do not contain vocals
    if "vocal" in key:
        continue

    dic = instruments[key]
    note_pitch = dic["pitch"]
    instrument_family_str = dic["instrument_family_str"]
    program_num = instrument_mapping[instrument_family_str]
    note_velocity = dic["velocity"]
    
    # NOTE: in this work, we only evaluate the onsets, and ignore the offsets
    # and for NSynth, the onset is always at the beginning of the audio
    # so we set the offset to be 0.0 and the duration to be 4.0 for all samples
    duration = 4

    mid_obj = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program_num)
    note = pretty_midi.Note(
            velocity=note_velocity, 
            pitch=note_pitch, 
            start=0.05,                 # add a small offset as our models perform better when there is an offset =.=
            end=duration + 0.05
    )
    inst.notes.append(note)
    mid_obj.instruments.append(inst)
    mid_obj.write(os.path.join(prefix_path, "midi", key + ".mid"))