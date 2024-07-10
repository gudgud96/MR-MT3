# MR-MT3

Code accompanying paper: [MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage](https://arxiv.org/pdf/2403.10024.pdf).

## Setup steps

First, you need to install requirements:
```
python3 -m pip install -r requirements.txt
```
Make sure `pip` is up-to-date, which should be able to [reduce backtracking](https://pip.pypa.io/en/stable/topics/dependency-resolution/#possible-ways-to-reduce-backtracking).

Next, you need to download the required datasets and postprocess them:

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

Refer to `train.sh` for a list of train commands corresponding to all of our experiments.

## Evaluation

You can download our pretrained models at: [https://huggingface.co/gudgud1014/MR-MT3/tree/main](https://huggingface.co/gudgud1014/MR-MT3/tree/main) to run inference.

Refer to `test.sh`, for a list of test commands corresponding to all of our experiments.

Basically, each command runs `test.py` that:
- transcribe MIDI files based on a given `eval.audio_dir`;
- compute multi-instrument F1 score w.r.t. the ground truth MIDI.


## License
MIT

## Citations
If you find our research useful, kindly cite us at:
```
@article{tan2024mr,
  title={MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage},
  author={Tan, Hao Hao and Cheuk, Kin Wai and Cho, Taemin and Liao, Wei-Hsiang and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2403.10024},
  year={2024}
}
```

## Credits
Huge shoutout to [@kunato](https://github.com/kunato) as we largely based our initial MT3 experiments on his implementation - [https://github.com/kunato/mt3-pytorch](https://github.com/kunato/mt3-pytorch).
