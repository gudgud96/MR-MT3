## Setup steps

1. Install requirements - `python3 -m pip install -r requirements.txt`. Make sure `pip` is up-to-date.

### For Slakh

1. Re-sample Slakh `.flac` to 16kHz - `python3 resample.py`.

2. Create the grouped stem version as ground truth instead of the existing `all_src.mid`. Some bass notes have octave errors - `python3 midi_script.py`.

3. `python3 tools/generate_inst_names.py`

### For ComMU

1. Download ComMU dataset - `https://github.com/POZAlabs/ComMU-code/tree/master/dataset`

2. `cd scripts/commu/` -> `./process_commu_dataset.sh`

### For NSynth

1. Download NSynth dataset validation split.

2. `cd scripts/nsynth/` -> `python3 convert_nsynth_json_to_midi.py`


## Training

### Training model MT3Net with dataset Slakh
```
python train.py num_epochs=1 devices=[0,1] model=MT3Net dataset=Slakh
```

### Training model MT3NetXL with dataset SlakhXL
```
python train.py num_epochs=1 devices=[0,1] model=MT3NetXL dataset=SlakhXL
```

### Training model MT3Pix2Seq with dataset SlakhPix2Seq
```
python train.py num_epochs=1 devices=[5] model=MT3NetPix2Seq dataset=SlakhPix2Seq
```

With hydra, config can be overwritten in the CLI.

### Training model MT3Net with dataset ComMU
```
python3 train.py --config-path "config" --config-name "config_commu"
```

## Evaluation

Refer to `test.sh`, which runs `test.py` that (1) transcribe MIDI files based on given `eval.audio_dir`; (2) compute multi-instrument F1 score.

For example:
```
python3 test.py \
    --config-path "config" \
    --config-name "config_commu" \
    path="../../../outputs/<date>/<time>/version_0/checkpoints/last.pt" \                   # needs to be a .pt / .pth file
    eval.exp_tag_name="commu_mt3" \                                                         # specify your own tag name
    eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/*.wav" \     # audio path for `glob.glob`
    hydra/job_logging=disabled    \
```