# ==== MT3 official checkpoint ==== #
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_f1_0.65" \
    model="MT3Net" \
    path="../../../pretrained/mt3.pth" \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.eval_first_n_examples=1 \
    eval.contiguous_inference=False \
    eval.use_tf_spectral_ops=True \
    +eval.load_weights_strict=False \
    # eval.eval_first_n_examples=1 \

# ==== exp_segmemV2_prev_context=0 ==== #
# NOTE: this is MR-MT3 with no memory block, which is basically
# MT3, but trained from scratch
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=0.ckpt" \
    model_segmem_length=0 \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    split_frame_length=2000 \
    # eval.eval_first_n_examples=1 \

# ==== exp_segmemV2_prev_context=32 ==== #
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=32.ckpt" \
    model_segmem_length=32 \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    split_frame_length=2000 \
    # eval.eval_first_n_examples=1 \

# ==== exp_segmemV2_prev_context=64 ==== #
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=64.ckpt" \
    model_segmem_length=64 \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    split_frame_length=2000 \
    # eval.eval_first_n_examples=1 \

# ==== exp_segmemV2_prev_context=32_prevaug_frame=3 ==== #
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=64_prevaug_frame\=3.ckpt" \
    model_segmem_length=64 \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    split_frame_length=2000 \
    # eval.eval_first_n_examples=1 \

# ==== exp_segmemV2_prev_context=32_prevaug_frame=8 ==== #
python3 test.py \
    --config-dir="config" \
    --config-name="config_slakh_segmem" \
    model="MT3NetSegMemV2WithPrev" \
    path="../../../pretrained/exp_segmemV2_prev_context\=64_prevaug_frame\=8.ckpt" \
    model_segmem_length=64 \
    eval.eval_dataset="Slakh" \
    eval.exp_tag_name="slakh_mt3_official" \
    eval.audio_dir="/data/slakh2100_flac_redux/test/*/mix_16k.wav" \
    hydra/job_logging=disabled \
    eval.is_sanity_check=True   \
    eval.contiguous_inference=True \
    eval.use_tf_spectral_ops=False \
    split_frame_length=2000 \
    # eval.eval_first_n_examples=1 \

# ======= test ComMU ======= #
python3 test.py \
    --config-path "config" \
    --config-name "config_commu" \
    path="../../../pretrained/commu_mt3.pt" \
    eval.eval_dataset="ComMU" \
    eval.exp_tag_name="commu_mt3" \
    eval.audio_dir="/data/datasets/ComMU/dataset_processed/commu_audio_v2/test/*.wav" \
    hydra/job_logging=disabled \

# ======= test NSynth ======= #
# This is to compare between the performance of MT3 trained on Slakh / ComMU, on NSynth dataset
python3 test.py \
    path="../../../pretrained/mt3.pth" \
    eval.eval_dataset="NSynth" \
    eval.exp_tag_name="slakh_mt3_on_nsynth" \
    eval.audio_dir="/data/nsynth-valid/audio/*.wav" \
    eval.midi_dir="/data/nsynth-valid/midi/" \
    hydra/job_logging=disabled \

python3 test.py \
    --config-path "config" \
    --config-name "config_commu" \
    path="../../../pretrained/commu_mt3.pt" \
    eval.eval_dataset="NSynth" \
    eval.exp_tag_name="commu_mt3_on_nsynth" \
    eval.audio_dir="/data/nsynth-valid/audio/*.wav" \
    eval.midi_dir="/data/nsynth-valid/midi/" \
    hydra/job_logging=disabled \