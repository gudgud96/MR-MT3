import glob
import os
import librosa
import numpy as np
import soundfile as sf
from functools import partial
import miditoolkit
import re

dir = sorted(glob.glob("loops_data/*"))

# mix audio
# for idx, item in enumerate(dir):
#     song_name = item.split("/")[1]
#     print(idx, song_name)
#     audios = glob.glob(os.path.join(item, "Audio Stems/*.wav"))
#     ys = map(partial(librosa.load, sr=22050), audios)
#     ys = list(map(lambda x: x[0], ys))
#     # ys[-1] = np.pad(ys[-1], (0, 1), "constant", constant_values=0)
#     ys = np.stack(ys)
#     mix = np.sum(ys, axis=0)
#     sf.write(os.path.join("loops_data", song_name, "mix.wav"), mix, 22050, "PCM_24") 

# mix midi
for idx, item in enumerate(dir):
    song_name = item.split("/")[1]
    bpm = int(re.search("[0-9]+ BPM", song_name).group(0).split(" ")[0])
    print(idx, song_name, bpm)
    midis = sorted(glob.glob(os.path.join(item, "MIDI/*.mid")))
    mid_obj = miditoolkit.MidiFile()
    mid_obj.tempo_changes = [miditoolkit.midi.containers.TempoChange(bpm, 0)]
    for mid_track in midis:
        sub_mid_obj = miditoolkit.MidiFile(mid_track)
        mid_obj.ticks_per_beat = sub_mid_obj.ticks_per_beat
        inst = sub_mid_obj.instruments[0]
        program_num = input(mid_track + ": ")
        inst.program = int(program_num)

        mid_obj.instruments.append(inst)
    
    mid_obj.dump(os.path.join("loops_data", song_name, "mix.mid"))
    # break



