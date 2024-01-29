# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_P2P_DISABLE=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2" \
#     dataset="Slakh" \
#     model_segmem_length=4 \

# train segmem v2 context = 0
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2" \
    dataset="Slakh" \
    model_segmem_length=0 \
    path="../../../outputs/2023-12-10/20-21-18/MT3NetSegMemV2_Slakh/version_0/checkpoints/last.ckpt" \

# train segmem v1 context = 0
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMem" \
#     dataset="Slakh" \
#     model_segmem_length=0 \

# train baseline
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_f1_0.65" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3Net" \
<<<<<<< Updated upstream
#     dataset="Slakh" \
=======
#     dataset="Slakh" \
#     split_frame_length=2000 \

# train segmem with MT3 split_frame + prev_frame
# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrev" \
#     split_frame_length=2000 \
#     model_segmem_length=0 \
#     trainer.check_val_every_n_epoch=1 \

# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrev" \
#     split_frame_length=2000 \
#     model_segmem_length=64 \
#     trainer.check_val_every_n_epoch=20 \

# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrevAugment" \
#     split_frame_length=2000 \
#     model_segmem_length=64 \
#     dataset_prev_augment_frames=8 \
#     trainer.check_val_every_n_epoch=20 \
#     path="../../../outputs/2024-01-12/15-03-59/MT3NetSegMemV2WithPrev_SlakhPrevAugment/version_0/checkpoints/last.ckpt" \

# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV2WithPrev" \
#     dataset="SlakhPrevAugment" \
#     dataset_use_tf_spectral_ops=False \
#     split_frame_length=2000 \
#     model_segmem_length=64 \
#     dataset_prev_augment_frames=3 \
#     trainer.check_val_every_n_epoch=20 \
#     path="../../../outputs/2024-01-21/10-33-08/MT3NetSegMemV2WithPrev_SlakhPrevAugment/version_0/checkpoints/last.ckpt" \


HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2WithPrevFineTune" \
    dataset="SlakhPrev" \
    dataset_use_tf_spectral_ops=True \
    dataset_is_randomize_tokens=False \
    split_frame_length=2000 \
    model_segmem_length=0 \
    trainer.check_val_every_n_epoch=20 \
    path="../../../pretrained/mt3.pth" \
    optim.lr=1e-5 \
    # dataset_prev_augment_frames=3 \


# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_f1_0.65" \
#     devices=[1] \
#     hydra/job_logging=disabled \
#     model="MT3Net" \
#     dataset="SlakhStemMix" \
#     dataset_use_tf_spectral_ops=False \
#     split_frame_length=2000 \
#     trainer.check_val_every_n_epoch=20 \

# HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
#     --config-path="config" \
#     --config-name="config_slakh_segmem" \
#     devices=[0,1] \
#     hydra/job_logging=disabled \
#     model="MT3NetSegMemV3" \
#     dataset="SlakhPrev" \
#     split_frame_length=2000 \
#     model_segmem_length=64 \
#     trainer.check_val_every_n_epoch=20 \
>>>>>>> Stashed changes
