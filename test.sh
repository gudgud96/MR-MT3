# ComMU
# python3 test.py \
#     --config-path "config" \
#     --config-name "config_commu" \
#     path="../../../pretrained/commu_mt3.pt" \
#     eval.eval_dataset="ComMU" \
#     eval.exp_tag_name="commu_mt3" \
#     eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/*.wav" \
#     hydra/job_logging=disabled \


# Slakh MT3 baseline
python3 test.py \
    path="../../../pretrained/mt3.pth" \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    eval.midi_dir="/data/slakh2100_flac_redux/test/" \
    eval.eval_first_n_examples=3 \
    hydra/job_logging=disabled \
    

# NSynth
# This is to compare between the performance of MT3 trained on Slakh / ComMU, on NSynth dataset
# python3 test.py \
#     path="../../../pretrained/mt3.pth" \
#     eval.eval_dataset="NSynth" \
#     eval.exp_tag_name="slakh_mt3_on_nsynth" \
#     eval.audio_dir="/data/nsynth-valid/audio/*.wav" \
#     eval.midi_dir="/data/nsynth-valid/midi/" \
#     hydra/job_logging=disabled \

# python3 test.py \
#     --config-path "config" \
#     --config-name "config_commu" \
#     path="../../../pretrained/commu_mt3.pt" \
#     eval.eval_dataset="NSynth" \
#     eval.exp_tag_name="commu_mt3_on_nsynth" \
#     eval.audio_dir="/data/nsynth-valid/audio/*.wav" \
#     eval.midi_dir="/data/nsynth-valid/midi/" \
#     hydra/job_logging=disabled \
