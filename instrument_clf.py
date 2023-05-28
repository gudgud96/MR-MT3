import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from inference import InferenceHandler
import torch
import glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import concurrent.futures
from threading import Thread
from multiprocessing import Queue
import librosa
import traceback
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dir = sorted(glob.glob("/data/slakh2100_flac_redux/validation/*/mix.flac"))
# dir = random.choices(dir, k=300)
handler = InferenceHandler('./pretrained', device=torch.device('cuda'))


"""
There was a freaking weird error: 
tensorflow.python.framework.errors_impl.InvalidArgumentError: 
{{function_node __wrapped__Reshape_device_/job:localhost/replica:0/task:0/device:GPU:0}} 
Input to reshape is a tensor with 1023 values, but the requested shape has 1 [Op:Reshape]

I suspect spectral_ops is not thread-safe.
""" 

def func(fname):
    audio, _ = librosa.load(fname, sr=16000, mono=True)
    inputs, frame_times = handler._preprocess(audio)
    inputs_tensor = torch.from_numpy(inputs)
    inputs_tensor, frame_times = handler._batching(inputs_tensor, frame_times, batch_size=16)
    results = []
    for idx, batch in enumerate(inputs_tensor):
        batch = batch.cuda()
        model_output = handler.model.generate(inputs=batch, max_length=1024, num_beams=1, do_sample=False,
                                    length_penalty=0.4, eos_token_id=handler.model.config.eos_token_id, 
                                    early_stopping=False, bad_words_ids=None,
                                    return_dict_in_generate=True, output_hidden_states=True, output_attentions=True)

        result = model_output.sequences
        result = handler._postprocess_batch(result)
        result = result.squeeze().reshape(-1)
        
        # encoder hidden state
        encoder_hidden_state = torch.stack(model_output.encoder_hidden_states, dim=0).squeeze()
        encoder_hidden_state = encoder_hidden_state.reshape(encoder_hidden_state.shape[0], -1, encoder_hidden_state.shape[-1])
        encoder_hidden_state = torch.mean(encoder_hidden_state, dim=1).cpu().detach().numpy()
        
        # decoder hidden state
        decoder_hidden_states = [torch.cat(k, dim=1).transpose(1, 0) for k in model_output.decoder_hidden_states]
        decoder_hidden_states = torch.cat(decoder_hidden_states, dim=1)
        decoder_hidden_states = torch.mean(decoder_hidden_states, dim=1).cpu().detach().numpy()
        
        # instruments
        results.append(result)
    
    results = np.concatenate(results, axis=-1).squeeze()
    if len(results) != 0:
        min_program_id, max_program_id = handler.codec.event_type_range('program')
        inst_idx = np.where((result >= min_program_id) & (result <= max_program_id))[0]
        insts = result[inst_idx] - min_program_id
        insts_vec = torch.zeros(128).cpu().detach().numpy()
        insts_vec[insts] = 1
        
        np.save("inst_test/validation/" + fname.split("/")[-2] + f"_enc.npy", encoder_hidden_state)
        np.save("inst_test/validation/" + fname.split("/")[-2] + f"_dec.npy", decoder_hidden_states)
        np.save("inst_test/validation/" + fname.split("/")[-2] + f"_ins.npy", insts_vec) 

pbar = tqdm(total=len(dir))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Start the load operations and mark each future with its URL
    future_to_fname = {executor.submit(func, fname): fname for fname in dir}
    for future in concurrent.futures.as_completed(future_to_fname):
        try:
            fname = future_to_fname[future]
            audio = future.result()
            pbar.update()
        except Exception as e:
            traceback.print_exc()