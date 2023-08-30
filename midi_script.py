import miditoolkit
import glob
import os
from tqdm import tqdm

"""
Use this for evaluation!!!
"""

midis = sorted(glob.glob("/data/slakh2100_flac_redux/test/*/MIDI/"))
for midi in tqdm(midis):
    stems = sorted(glob.glob(midi + "*.mid"))
    insts = []
    for stem in stems:
        midi_obj = miditoolkit.MidiFile(stem)
        for inst in midi_obj.instruments:
            insts.append(inst)

    new_midi_obj = miditoolkit.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
    new_midi_obj.tempo_changes = midi_obj.tempo_changes
    new_midi_obj.key_signature_changes = midi_obj.key_signature_changes
    new_midi_obj.instruments = insts

    new_midi_obj.dump(os.path.join(midi.replace("MIDI/", ""), "all_src_v2.mid"))