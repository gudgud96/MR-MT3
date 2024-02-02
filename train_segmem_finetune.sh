# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem_finetune" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrevFineTune" \
#     dataset="SlakhPrev" \
#     dataset_use_tf_spectral_ops=True \
#     dataset_is_randomize_tokens=True \
#     split_frame_length=2000 \
#     model_segmem_length=0 \
#     trainer.check_val_every_n_epoch=20 \
#     optim.lr=1e-5 \
#     num_epochs=100 \
#     path="../../../pretrained/mt3.pth" \

HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem_finetune" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2WithPrevFineTune" \
    dataset="SlakhPrevAugment" \
    dataset_use_tf_spectral_ops=True \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    dataset_prev_augment_frames=3 \
    trainer.check_val_every_n_epoch=20 \
    optim.lr=1e-5 \
    num_epochs=100 \
    path="../../../pretrained/mt3.pth" \