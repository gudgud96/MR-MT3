## NSynth evaluation

Instead of MIDI-based evaluation, this evaluation analyze instrument and pitch accuracy for NSynth.

1. `python3 parse_nsynth_valid.py --tag_name <create_your_own> --path <midi_path>`

2. `python3 evaluate_nsynth_json.py --json_path <create_your_own>.json`

ComMU:

```
Onset + program F1 (flat): 0.3556
Onset + program F1 (full): 0.03011
Onset + program F1 (midi_class): 0.09417
Onset + program precision (flat): 0.325
Onset + program precision (full): 0.02691
Onset + program precision (midi_class): 0.08913
Onset + program recall (flat): 0.4481
Onset + program recall (full): 0.03781
Onset + program recall (midi_class): 0.1094
Onset F1: 0.3556
Onset precision: 0.325
Onset recall: 0.4481

Instrument acc: 19.11%
Pitch acc: 48.26%
Avg num tracks: 0.87
Avg num instruments after MIDI grouping: 0.86
```

Slakh:

```
Onset + program F1 (flat): 0.415
Onset + program F1 (full): 0.08731
Onset + program F1 (midi_class): 0.2059
Onset + program precision (flat): 0.3828
Onset + program precision (full): 0.08247
Onset + program precision (midi_class): 0.1934
Onset + program recall (flat): 0.5205
Onset + program recall (full): 0.1021
Onset + program recall (midi_class): 0.2411
Onset F1: 0.4153
Onset precision: 0.383
Onset recall: 0.5211

Instrument acc: 41.37%
Pitch acc: 48.88%
Avg num instruments: 1.04
Avg num tracks: 1.08
```