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
#     dataset="Slakh" \