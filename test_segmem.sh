# ==== exp_segmemV2_prev_context=0 ==== #
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