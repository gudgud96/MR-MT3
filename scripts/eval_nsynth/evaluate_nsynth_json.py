import json
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str)
args = parser.parse_args()

with open(args.json_path) as f:
    nsynth_valid_results = json.load(f)

correct_instrument, correct_pitch, avg_instrument, avg_tracks = 0, 0, 0, 0
for key in nsynth_valid_results:
    elem = nsynth_valid_results[key]
    num_instrument = elem["predicted"]["num_instruments"]
    avg_instrument += num_instrument

    expected_instrument = elem["expected_instrument"]
    predicted = elem["predicted"]["events"]
    if len(predicted) == 0:
        continue
    predicted_instrument = predicted[0]["instrument"]
    if expected_instrument == predicted_instrument:
        correct_instrument += 1
    
    expected_pitch = elem["expected_pitch"]
    predicted_pitch = elem["predicted"]["events"][0]["pitch"]
    if expected_pitch in predicted_pitch:
        correct_pitch += 1
    
    avg_tracks += elem["num_tracks"]

print(f"Instrument acc: {correct_instrument / len(nsynth_valid_results) * 100:.2f}%")
print(f"Pitch acc: {correct_pitch / len(nsynth_valid_results) * 100:.2f}%")
print(f"Avg num tracks: {avg_tracks / len(nsynth_valid_results):.2f}")
print(f"Avg num instruments after MIDI grouping: {avg_instrument / len(nsynth_valid_results):.2f}")
