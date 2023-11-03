## Setup steps

1. Install requirements - `python3 -m pip install -r requirements.txt`. Make sure `pip` is up-to-date.

### For Slakh

1. Re-sample Slakh `.flac` to 16kHz - `python3 resample.py`.

2. Create the grouped stem version as ground truth instead of the existing `all_src.mid`. Some bass notes have octave errors - `python3 midi_script.py`.

3. `python3 tools/generate_inst_names.py`

### For ComMU

1. Download ComMU dataset - `https://github.com/POZAlabs/ComMU-code/tree/master/dataset`

2. `cd scripts` -> `./process_commu_dataset.sh`


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
python3 train.py \
    num_epochs=400 \
    mel_length=256 \
    optim.lr=2e-4 \
    optim.min_lr=1e-4 \
    optim.num_steps_per_epoch=5289 \
    optim.warmup_steps=64500 \ 
    devices=[0,1] \
    model=MT3Net    \
    dataset=ComMU
```
NOTE: Can we use a separate config file?

After training is done, run `python3 mt3_net.py mode=test path=<checkpoint_path.ckpt>` to convert `.ckpt` to `.pth`.

## Evaluation
`python3 analysis.py` -> `python3 evaluate.py`