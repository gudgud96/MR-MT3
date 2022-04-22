import json
import torch
import numpy as np
import torch.nn as nn


def _map_state_dict(state_dict, mapping):
    import jax
    import jax.numpy as jnp
    new_state_dict = {**state_dict}
    pt_model_dict = {}
    remove_key = []
    for k, v in mapping.items():
        value = state_dict[v]
        remove_key.append(v)
        if k.endswith("kernel") and value.ndim == 4 and k not in pt_model_dict:
            k = k.replace('kernel', 'weight')
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            value = jnp.transpose(value, (3, 2, 0, 1))
        elif k.endswith("kernel") and k not in pt_model_dict:
            # linear layer
            k = k.replace('kernel', 'weight')
            value = value.T
        elif k.endswith("scale") or k.endswith('embedding'):
            k = k.replace('kernel', 'weight')
        new_state_dict[k] = torch.from_numpy(value)
    for k in list(set(remove_key)):
        del new_state_dict[k]
    return new_state_dict


def convert_t5x_to_pt(config, flatten_statedict):

    state_dict_mapping = {}

    # Encoder
    for layer_index in range(config['num_layers']):
        layer_name = f"layers_{str(layer_index)}"

        encoder_layer_state_dict = {
            f"encoder.block.{str(layer_index)}.layer.0.SelfAttention.k.kernel": f'target/encoder/{layer_name}/attention/key/kernel',
            f"encoder.block.{str(layer_index)}.layer.0.SelfAttention.o.kernel": f'target/encoder/{layer_name}/attention/out/kernel',
            f"encoder.block.{str(layer_index)}.layer.0.SelfAttention.q.kernel": f'target/encoder/{layer_name}/attention/query/kernel',
            f"encoder.block.{str(layer_index)}.layer.0.SelfAttention.v.kernel": f'target/encoder/{layer_name}/attention/value/kernel',
            f"encoder.block.{str(layer_index)}.layer.0.layer_norm.weight": f'target/encoder/{layer_name}/pre_attention_layer_norm/scale',

            f"encoder.block.{str(layer_index)}.layer.1.DenseReluDense.wi_0.kernel": f"target/encoder/{layer_name}/mlp/wi_0/kernel",
            f"encoder.block.{str(layer_index)}.layer.1.DenseReluDense.wi_1.kernel": f"target/encoder/{layer_name}/mlp/wi_1/kernel",
            f"encoder.block.{str(layer_index)}.layer.1.DenseReluDense.wo.kernel": f"target/encoder/{layer_name}/mlp/wo/kernel",
            f"encoder.block.{str(layer_index)}.layer.1.layer_norm.weight": f"target/encoder/{layer_name}/pre_mlp_layer_norm/scale",
        }
        state_dict_mapping.update(encoder_layer_state_dict)

    # Decoder
    for layer_index in range(config['num_layers']):
        layer_name = f"layers_{str(layer_index)}"

        decoder_layer_state_dict = {
            f"decoder.block.{str(layer_index)}.layer.0.SelfAttention.k.kernel": f'target/decoder/{layer_name}/self_attention/key/kernel',
            f"decoder.block.{str(layer_index)}.layer.0.SelfAttention.o.kernel": f'target/decoder/{layer_name}/self_attention/out/kernel',
            f"decoder.block.{str(layer_index)}.layer.0.SelfAttention.q.kernel": f'target/decoder/{layer_name}/self_attention/query/kernel',
            f"decoder.block.{str(layer_index)}.layer.0.SelfAttention.v.kernel": f'target/decoder/{layer_name}/self_attention/value/kernel',

            f'decoder.block.{str(layer_index)}.layer.0.layer_norm.weight': f"target/decoder/{layer_name}/pre_self_attention_layer_norm/scale",


            f"decoder.block.{str(layer_index)}.layer.1.EncDecAttention.k.kernel": f'target/decoder/{layer_name}/encoder_decoder_attention/key/kernel',
            f"decoder.block.{str(layer_index)}.layer.1.EncDecAttention.o.kernel": f'target/decoder/{layer_name}/encoder_decoder_attention/out/kernel',
            f"decoder.block.{str(layer_index)}.layer.1.EncDecAttention.q.kernel": f'target/decoder/{layer_name}/encoder_decoder_attention/query/kernel',
            f"decoder.block.{str(layer_index)}.layer.1.EncDecAttention.v.kernel": f'target/decoder/{layer_name}/encoder_decoder_attention/value/kernel',

            f"decoder.block.{str(layer_index)}.layer.1.layer_norm.weight": f'target/decoder/{layer_name}/pre_cross_attention_layer_norm/scale',


            f"decoder.block.{str(layer_index)}.layer.2.DenseReluDense.wi_0.kernel": f"target/decoder/{layer_name}/mlp/wi_0/kernel",
            f"decoder.block.{str(layer_index)}.layer.2.DenseReluDense.wi_1.kernel": f"target/decoder/{layer_name}/mlp/wi_1/kernel",
            f"decoder.block.{str(layer_index)}.layer.2.DenseReluDense.wo.kernel": f"target/decoder/{layer_name}/mlp/wo/kernel",
            f"decoder.block.{str(layer_index)}.layer.2.layer_norm.weight": f'target/decoder/{layer_name}/pre_mlp_layer_norm/scale',

        }
        state_dict_mapping.update(decoder_layer_state_dict)

    generic_state_dict = {
        "lm_head.kernel": "target/decoder/logits_dense/kernel",
        "encoder.final_layer_norm.weight": "target/encoder/encoder_norm/scale",
        "decoder.final_layer_norm.weight": "target/decoder/decoder_norm/scale",
        "decoder.embed_tokens.weight": "target/decoder/token_embedder/embedding",
        "decoder_embed_tokens.weight": "target/decoder/token_embedder/embedding",
        "proj.kernel": 'target/encoder/continuous_inputs_projection/kernel',
        "encoder.embed_tokens.kernel": 'target/encoder/continuous_inputs_projection/kernel',
    }
    state_dict_mapping.update(generic_state_dict)
    pt_state_dict = _map_state_dict(
        flatten_statedict, mapping=state_dict_mapping)
    assert np.allclose(pt_state_dict['proj.weight'].numpy(
    ).T, flatten_statedict['target/encoder/continuous_inputs_projection/kernel'])
    return pt_state_dict


def parse_t5x_state_dict(state_dict):
    from t5x import state_utils
    flatten_all_statedict = (
        state_utils.flatten_state_dict(
            state_dict, keep_empty_nodes=True))
    flatten_statedict = {}

    for k, v in flatten_all_statedict.items():
        if not k.startswith('state'):
            flatten_statedict[k] = v
    return flatten_statedict


def load_t5x_statedict(path='/home/kunato/mt3/mt3_flax_state_dict.pk'):
    """
    ### TO SAVE STATE_DICT of t5x
    #### Add this to  (t5x/t5x/checkpoints.py) (line 950) (def restore())
    #### print('ckpt_state_dict', state_utils.flatten_state_dict(
    ####     state_dict).keys())

    #### pickle.dump(state_dict, open('mt3_flax_state_dict.pk', 'wb'))

    """
    import pickle
    state_dict = pickle.load(open(path, 'rb'))
    return state_dict


def main():
    state_dict = load_t5x_statedict()
    state_dict = parse_t5x_state_dict(state_dict)
    config_dict = {
        "architectures": [
            "T5ForConditionalGeneration"
        ],
        "d_ff": 1024,
        "d_kv": 64,
        "d_model": 512,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "pad_token_id": 0,  # -2
        "eos_token_id": 1,
        "unk_token_id": 2,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "num_heads": 6,
        "num_decoder_layers": 8,
        "num_layers": 8,
        "output_past": True,
        "tie_word_embeddings": False,
        "vocab_size": 1536
    }
    state_dict = convert_t5x_to_pt(
        config=config_dict, flatten_statedict=state_dict)
    from models.t5 import T5ForConditionalGeneration, T5Config
    config = T5Config.from_dict(config_dict)
    model: nn.Module = T5ForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    torch.save(model.state_dict(), 'pretrained/mt3.pth')
    with open('pretrained/config.json', 'w') as w:
        json.dump(config_dict, w)


if __name__ == '__main__':
    main()
