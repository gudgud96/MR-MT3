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
# python3 test.py \
#     path="/data2/kinwai/mt3_clean/outputs/2023-11-10/22-54-57/MT3NetCTC_Slakh/version_0/checkpoints/best.ckpt" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data2/kinwai/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     eval.is_sanity_check=False \
#     hydra/job_logging=disabled \

# DETR
python3 test.py \
    --config-name "config_slakh_newtoken" \
    path="/data2/kinwai/mt3_clean/outputs/2023-12-11/15-13-27/DETR_SlakhNew/version_0/checkpoints/76-49665-27.5033.ckpt" \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_detr" \
    eval.audio_dir="/data2/kinwai/slakh2100_flac_redux/test/*/mix_16k.wav" \
    eval.is_sanity_check=False \
    hydra/job_logging=disabled \