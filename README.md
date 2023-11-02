## Setup steps

1. Install requirements - `python3 -m pip install -r requirements.txt`. Make sure `pip` is up-to-date.

2. Re-sample Slakh `.flac` to 16kHz - `python3 resample.py`.

3. Create the grouped stem version as ground truth instead of the existing `all_src.mid`. Some bass notes have octave errors - `python3 midi_script.py`.

4. `python3 tools/generate_inst_names.py`

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

with hydra, config can be overwritten in the CLI.

After training is done, run `python3 mt3_net.py mode=test path=<checkpoint_path.ckpt>` to convert `.ckpt` to `.pth`.

## Evaluation
`python3 analysis.py` -> `python3 evaluate.py`