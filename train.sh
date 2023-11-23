# path="../../../outputs/2023-11-17/00-52-56/MT3Net_SlakhStemMix/version_0/checkpoints/last.ckpt" \

python3 train.py \
    --config-path config \
    --config-name config_slakh_segmem \
    devices=[0,1] \
    model="MT3NetSegMem" \
    dataset="Slakh" \
    # path="../../../outputs/2023-11-22/23-43-18/MT3NetSegMem_ComMU/version_0/checkpoints/last.ckpt" \

# python3 train.py \
#     --config-path config \
#     --config-name config_slakh_segmem \
#     devices=[0] \
#     model="MT3Net" \
#     dataset="ComMU" \
#     path="../../../outputs/2023-11-22/22-16-25/MT3Net_ComMU/version_0/checkpoints/last.ckpt" \