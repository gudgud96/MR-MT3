HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_f1_0.65" \
    devices=[1] \
    hydra/job_logging=disabled \
    model="MT3Net" \
    dataset="SlakhStemMix" \
    dataset_use_tf_spectral_ops=False \
    split_frame_length=2000 \
    trainer.check_val_every_n_epoch=20 \