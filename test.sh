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
#     --config-dir="config" \
#     --config-name="config" \
#     model="MT3Net" \
#     path="../../../pretrained/mt3.pth" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True   \
#     eval.eval_first_n_examples=1 \
#     eval.contiguous_inference=False \

# MT3 pretrained, but load weights for segmem
# path="../../../pretrained/mt3.pth" \
# path="../../../outputs/2024-01-28/15-38-25/MT3NetSegMemV2WithPrevFineTune_SlakhPrev/version_0/checkpoints/last.ckpt" \

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=64_prevaug_frame\=8.ckpt" \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    model_segmem_length=64 \
    split_frame_length=2000 \
    eval.eval_first_n_examples=1 \
    +eval.load_weights_strict=False \

# python3 test.py \
#     --config-dir="config" \
#     --config-name="config_slakh_f1_0.65" \
#     model="MT3Net" \
#     path="../../../outputs/2023-11-24/11-15-55/MT3Net_Slakh/version_0/checkpoints/epoch\=379-step\=245100-val_loss\=1.3445.ckpt" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True   \
#     eval.contiguous_inference=False \
#     eval.use_tf_spectral_ops=False \
    # eval.eval_first_n_examples=1 \

# python3 test.py \
#     --config-dir="config" \
#     --config-name="config_slakh_segmem" \
#     model="MT3NetSegMem" \
#     path="../../../outputs/2023-12-08/23-56-30/MT3NetSegMem_Slakh/version_0/checkpoints/epoch\=399-step\=258000-val_loss\=1.4177.ckpt" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True   \
#     eval.contiguous_inference=False \
#     model_segmem_length=0 \
#     eval.eval_first_n_examples=1 \
#     eval.use_tf_spectral_ops=False \


# python3 test.py \
#     --config-dir="config" \
#     --config-name="config_slakh_segmem" \
#     model="MT3NetSegMemV2" \
#     path="../../../outputs/2023-11-30/18-13-37/MT3NetSegMemV2_Slakh/version_0/checkpoints/epoch\=399-step\=258000-val_loss\=1.2660.ckpt" \
#     eval.eval_dataset="Slakh" \
#     eval.exp_tag_name="slakh_mt3_official" \
#     eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
#     hydra/job_logging=disabled \
#     eval.is_sanity_check=True   \
#     eval.contiguous_inference=True \
#     model_segmem_length=16 \
#     eval.eval_first_n_examples=1 \

    

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