# ======= train baseline ======= #
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_f1_0.65" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3Net" \
#     dataset="Slakh" \
#     split_frame_length=2000 \

# ======= train segmem v2 context = N ======= #
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2" \
#     dataset="Slakh" \
#     dataset_use_tf_spectral_ops=False \
#     dataset_is_randomize_tokens=True \
#     split_frame_length=2000 \
#     model_segmem_length=0 \
#     trainer.check_val_every_n_epoch=20 \

#  ======= train segmem with prev_frame and context = N  ======= #
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrev" \
#     dataset_use_tf_spectral_ops=False \
#     dataset_is_randomize_tokens=True \
#     split_frame_length=2000 \
#     model_segmem_length=0 \
#     trainer.check_val_every_n_epoch=20 \

#  ======= train segmem with prev_frame, prev_augment, context = N  ======= #
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrevAugment" \
#     dataset_use_tf_spectral_ops=False \
#     dataset_is_randomize_tokens=True \
#     split_frame_length=2000 \
#     model_segmem_length=64 \
#     dataset_prev_augment_frames=8 \
#     trainer.check_val_every_n_epoch=20 \