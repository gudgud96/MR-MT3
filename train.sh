#  ======= train baseline ======= #
#  This experiment trains MT3 from scratch
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_f1_0.65" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3Net" \
    dataset="Slakh" \
    split_frame_length=2000 \
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=False \

#  ======= train segmem with prev_frame and context = N  ======= #
#  This experiment trains MR-MT3 which takes the immediate previous segment
#  as memory. The memory block is truncated at length `model_segmem_length`
#  which corresponds to `L_agg` in the paper.
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2WithPrev" \
    dataset="SlakhPrev" \
    dataset_use_tf_spectral_ops=False \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    trainer.check_val_every_n_epoch=20 \
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=True \

#  ======= train segmem with prev_frame, prev_augment, context = N  ======= #
#  This experiment trains MR-MT3 which takes the prior segment as memory.
#  This prior segment can be up to N "hops" before the current segment, where 
#  N = `dataset_prev_augment_frames`, and is written as `L_max_hop` in the paper. 
#  Similarly, the memory block is truncated at length `model_segmem_length`
#  which corresponds to `L_agg` in the paper.
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="MT3NetSegMemV2WithPrev" \
    dataset="SlakhPrevAugment" \
    dataset_use_tf_spectral_ops=False \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    dataset_prev_augment_frames=3 \
    trainer.check_val_every_n_epoch=20 \
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=True \

#  ======= continual training  ======= #
#  This experiment pre-loads MT3 official checkpoint, and continue training for N epochs
#  with the experiment settings proposed above.
#  Note that following MT3 official checkpoint, we need to use TF spectral_ops.
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
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=True \

#  ======= train vanilla Transformer baseline ======= #
#  This experiment trains vanilla Transformer from scratch
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_f1_0.65" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="VanillaTransformerNet" \
    dataset="Slakh" \
    split_frame_length=2000 \
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=False \

#  ======= train vanilla Transformer segmem with prev_frame and context = N  ======= #
#  This experiment trains vanilla Transformer which takes the immediate previous segment
#  as memory. The memory block is truncated at length `model_segmem_length`
#  which corresponds to `L_agg` in the paper.
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 python3 train.py \
    --config-path="config" \
    --config-name="config_slakh_segmem" \
    devices=[0,1] \
    hydra/job_logging=disabled \
    model="VanillaTransformerNetSegMemV2WithPrev" \
    dataset="SlakhPrev" \
    dataset_use_tf_spectral_ops=False \
    dataset_is_randomize_tokens=True \
    split_frame_length=2000 \
    model_segmem_length=64 \
    trainer.check_val_every_n_epoch=20 \
    eval.eval_after_num_epoch=400 \
    eval.eval_first_n_examples=3 \
    eval.eval_per_epoch=10 \
    eval.contiguous_inference=True \