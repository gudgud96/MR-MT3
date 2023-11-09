# ComMU
python3 test.py \
    --config-path "config" \
    --config-name "config_commu" \
    path="../../../pretrained/commu_mt3.pt" \
    eval.eval_dataset="ComMU" \
    eval.exp_tag_name="commu_mt3" \
    eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/*.wav" \
    hydra/job_logging=disabled \


# Slakh MT3 baseline
# python3 test.py \
#     path="../../../pretrained/mt3.pth" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     eval.is_sanity_check=True \
#     hydra/job_logging=disabled \