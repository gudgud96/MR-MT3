# ComMU
# CUDA_VISIBLE_DEVICES=1 python3 test.py \
#     --config-path "config" \
#     --config-name "config_commu" \
#     path="../../../pretrained/mt3.pth" \
#     eval.eval_dataset="ComMU" \
#     eval.exp_tag_name="commu_mt3" \
#     eval.eval_first_n_examples=10 \
#     eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/*.wav" \
#     hydra/job_logging=disabled \

# CUDA_VISIBLE_DEVICES=0 python3 test.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     path="../../../outputs/2023-11-21/23-09-37/MT3NetSegMem_ComMU/version_0/checkpoints/last.ckpt" \
#     model="MT3NetSegMem" \
#     eval.eval_dataset="ComMU" \
#     eval.exp_tag_name="commu_mt3" \
#     eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/train/*.wav" \
#     eval.eval_first_n_examples=1 \
#     eval.contiguous_inference=False \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True \
#     eval.batch_size=9 \


CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    path="../../../outputs/2023-11-23/00-42-58/MT3NetSegMem_Slakh/version_0/checkpoints/last.ckpt" \
    model="MT3NetSegMem" \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    eval.midi_dir="/data/slakh2100_flac_redux/test/" \
    eval.eval_first_n_examples=10 \
    eval.contiguous_inference=False \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True \
    eval.batch_size=8 \


# CUDA_VISIBLE_DEVICES=0 python3 test.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     path="../../../pretrained/mt3.pth" \
#     model="MT3Net" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     eval.midi_dir="/data/slakh2100_flac_redux/test/" \
#     eval.eval_first_n_examples=1 \
#     eval.contiguous_inference=False \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True \

# path="../../../outputs/2023-11-17/22-03-51/MT3Net_SlakhStemMix/version_0/checkpoints/last.ckpt" \

# Slakh MT3 baseline
# CUDA_VISIBLE_DEVICES=0 python3 test.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     path="../../../outputs/2023-11-21/18-00-33/MT3NetSegMem_Slakh/version_0/checkpoints/last.ckpt" \
#     model="MT3NetSegMem" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/train/*/mix_16k.wav" \
#     eval.midi_dir="/data/slakh2100_flac_redux/train/" \
#     eval.eval_first_n_examples=1 \
#     eval.contiguous_inference=True \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True \

# CUDA_VISIBLE_DEVICES=0 python3 test.py \
#     --config-path="config" \
#     --config-name="config_slakh_f1_0.65" \
#     path="../../../outputs/2023-11-19/22-46-29/MT3NetSegMem_Slakh/version_0/checkpoints/last.ckpt" \
#     model="MT3NetSegMem" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/train/*/mix_16k.wav" \
#     eval.midi_dir="/data/slakh2100_flac_redux/train/" \
#     eval.eval_first_n_examples=1 \
#     hydra/job_logging=disabled \
#     eval.contiguous_inference=True \
#     eval.is_sanity_check=True \
    

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