HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2WithPrev" \
    dataset="SlakhStemMixPrevAugment" \
    dataset_use_tf_spectral_ops=False \
    split_frame_length=2000 \
    model_segmem_length=64 \
    dataset_prev_augment_frames=3 \
    trainer.check_val_every_n_epoch=20 \