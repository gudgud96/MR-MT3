import json
import math
import os
from typing import List
import pickle

import numpy as np
from tqdm import tqdm
from models.t5 import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch
import librosa
from contrib import spectrograms, vocabularies, note_sequences, metrics_utils
import note_seq
import matplotlib.pyplot as plt
import traceback
import zlib


MIN_LOG_MEL = -12
MAX_LOG_MEL = 5


class InferenceHandler:

    def __init__(self, root_path, weight_path, device=torch.device('cuda')) -> None:
        config_path = f'{root_path}/config.json'
        with open(config_path) as f:
            config_dict = json.load(f)
        config = T5Config.from_dict(config_dict)
        model: nn.Module = T5ForConditionalGeneration(config)
        model.load_state_dict(torch.load(
            weight_path, map_location='cpu'), strict=True)
        model.eval()
        self.SAMPLE_RATE = 16000
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=1))
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        self.device = device
        print("device", self.device)
        self.model = model
        self.model.to(self.device)

    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        spectrogram_config = self.spectrogram_config
        frame_size = spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        print('audio', audio.shape, 'frame_size', frame_size)
        frames = spectrograms.split_audio(audio, spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / \
            spectrogram_config.frames_per_second
        return frames, times

    def _split_token_into_length(self, frames, frame_times, max_length=256):
        assert len(frames.shape) >= 1
        assert frames.shape[0] == frame_times.shape[0]
        num_segment = math.ceil(frames.shape[0] / max_length)
        batchs = []
        frame_times_batchs = []
        paddings = []
        for i in range(num_segment):
            batch = np.zeros((max_length, *frames.shape[1:]))
            frame_times_batch = np.zeros((max_length))
            start_idx = i * max_length
            end_idx = max_length if start_idx + \
                max_length < frames.shape[0] else frames.shape[0] - start_idx
            batch[0: end_idx, ...] = frames[start_idx: start_idx + end_idx, ...]
            frame_times_batch[0:end_idx] = frame_times[start_idx:start_idx + end_idx]
            batchs.append(batch)
            frame_times_batchs.append(frame_times_batch)
            paddings.append(end_idx)
        return np.stack(batchs, axis=0), np.stack(frame_times_batchs, axis=0), paddings

    def _compute_spectrograms(self, inputs):
        outputs = []
        outputs_raws = []
        for i in inputs:
            samples = spectrograms.flatten_frames(i)
            i = spectrograms.compute_spectrogram(
                samples, self.spectrogram_config)
            raw_i = samples
            outputs.append(i)
            outputs_raws.append(raw_i)
        
        melspec, raw = np.stack(outputs, axis=0), np.stack(outputs_raws, axis=0)

        # add normalization
        melspec = np.clip(melspec, MIN_LOG_MEL, MAX_LOG_MEL)
        melspec = (melspec - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
        return melspec, raw

    def _preprocess(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        frames, frame_times, paddings = self._split_token_into_length(
            frames, frame_times)
        inputs, _ = self._compute_spectrograms(frames)
        for i, p in enumerate(paddings):
            inputs[i, p:] = 0
        return inputs, frame_times

    def _batching(self, tensors, frame_times, batch_size=5):
        batchs = []
        frame_times_batch = []
        for start_idx in range(0, tensors.shape[0], batch_size):
            end_idx = min(start_idx+batch_size, tensors.shape[0])
            batchs.append(tensors[start_idx: end_idx])
            frame_times_batch.append(frame_times[start_idx: end_idx])
        return batchs, frame_times_batch

    def _get_program_ids(self, valid_programs) -> List[List[int]]:
        min_program_id, max_program_id = self.codec.event_type_range('program')
        total_programs = max_program_id - min_program_id
        invalid_programs = []
        for p in range(total_programs):
            if p not in valid_programs:
                invalid_programs.append(p)
        invalid_programs = [min_program_id + id for id in invalid_programs]
        invalid_programs = self.vocab.encode(invalid_programs)
        return [[p] for p in invalid_programs]

    # TODO Force generate using subset of instrument instead of all.

    @torch.no_grad()
    def get_encoder_outputs(self, audio_path, valid_programs=None):
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        if valid_programs is not None:
            invalid_programs = self._get_program_ids(valid_programs)
        else:
            invalid_programs = None
        inputs, frame_times = self._preprocess(audio)
        inputs_tensor = torch.from_numpy(inputs)
        results = []
        inputs_tensor, frame_times = self._batching(inputs_tensor, frame_times)
        for idx, batch in tqdm(enumerate(inputs_tensor), total=len(inputs_tensor)):
            batch = batch.to(self.device)
            encoder_outputs = self.model.encoder(input_ids=batch)
            hidden_states = encoder_outputs[0]
            results.append(hidden_states)
        
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def inference(self, audio, audio_path, outpath=None, valid_programs=None, num_beams=1, batch_size=5):
        try:
            if valid_programs is not None:
                invalid_programs = self._get_program_ids(valid_programs)
            else:
                invalid_programs = None
            # print('preprocessing', audio_path)
            inputs, frame_times = self._preprocess(audio)
            inputs_tensor = torch.from_numpy(inputs)
            saved_outputs = []
            results = []
            # print('batching', audio_path)
            inputs_tensor, frame_times = self._batching(inputs_tensor, frame_times, batch_size=batch_size)
            instruments = set()
            print('inferencing', audio_path)

            seqs, decs = [], []
            self.model.cuda()
            for idx, batch in enumerate(inputs_tensor):
                batch = batch.to(self.device)

                result = self.model.generate(inputs=batch, max_length=1024, num_beams=num_beams, do_sample=False,
                                            length_penalty=0.4, eos_token_id=self.model.config.eos_token_id, 
                                            early_stopping=False, bad_words_ids=invalid_programs,
                                            use_cache=True,
                                            # return_dict_in_generate=True, 
                                            # output_hidden_states=True,
                                            # output_attentions=True
                                            )
                
                # print('hidden', len(model_output.encoder_hidden_states), len(model_output.decoder_hidden_states))
                # print('attention', len(model_output.encoder_attentions), len(model_output.decoder_attentions))
                # result = model_output.sequences
                # result = self.model.generate(inputs=batch, max_length=1024, num_beams=num_beams, do_sample=False,
                #                          length_penalty=0.4, eos_token_id=self.model.config.eos_token_id, early_stopping=False, bad_words_ids=invalid_programs)

                result = self._postprocess_batch(result)
                # single_event = self._to_event([result], [frame_times[idx]])

                # play something easy first, instrument existence
                # instruments.update(set([note.program for note in single_event.notes if not note.is_drum]))

                results.append(result)

                # dec = torch.stack([torch.stack(k) for k in model_output.decoder_hidden_states])[:, -1, :, :, :].squeeze()
                # dec = torch.permute(dec, (1, 0, 2))

                # enc = []
                # for hid in model_output.encoder_hidden_states:
                #     tmp = hid.cpu().detach()
                #     enc.append(tmp)
                # enc = torch.stack(enc).squeeze()
                # # enc = torch.mean(enc, dim=2)    # to save space
                # enc = enc.numpy()
                
                # dec = []
                # for hid in model_output.decoder_hidden_states:
                #     tmp = torch.stack(hid).squeeze()
                #     tmp = tmp.cpu().detach()
                #     dec.append(tmp)
                # dec = torch.stack(dec).squeeze()
                # dec = torch.permute(dec, (1, 2, 0, 3))
                # # dec = torch.mean(dec, dim=2)    # to save space
                # dec = dec.numpy()

                # print(enc.shape, dec.shape)

                # seqs.append(result)
                # decs.append(dec.cpu().detach().numpy())

                # saved = {
                #     "input": batch.cpu().detach().numpy(),
                #     "output": model_output,
                #     "event": single_event
                # }
                # saved_outputs.append(saved)
            
            event = self._to_event(results, frame_times)
            if outpath is None:
                filename = audio_path.split('/')[-1].split('.')[0]
                outpath = f'./out/{filename}.mid'
            os.makedirs('/'.join(outpath.split('/')[:-1]), exist_ok=True)
            print("saving", outpath)
            note_seq.sequence_proto_to_midi_file(event, outpath)

            # seqs = np.concatenate(seqs, axis=1)
            # decs = np.concatenate(decs, axis=1)

            # print('seqss', seqs.shape, decs.shape)

            # np.save(outpath.replace("mix.mid", "seqs.npy"), seqs)
            # np.save(outpath.replace("mix.mid", "decs.npy"), decs)

            # with open(outpath.replace("mid", "pickle"), 'wb') as f:
            #     pickle.dump(saved_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        except Exception as e:
            traceback.print_exc()

    
    def load_inputs_tensor(self, audio_path, valid_programs=None, num_beams=1):
        print('loading', audio_path)
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        if valid_programs is not None:
            invalid_programs = self._get_program_ids(valid_programs)
        else:
            invalid_programs = None
        print('preprocessing', audio_path)
        inputs, frame_times = self._preprocess(audio)
        inputs_tensor = torch.from_numpy(inputs)
        print('batching', audio_path)
        inputs_tensor, frame_times = self._batching(inputs_tensor, frame_times, batch_size=128)

        return inputs_tensor, frame_times

    @torch.no_grad()
    def model_inference(self, inputs_tensor, frame_times, num_beams=1):
        encoder_outputs = []
        instruments = set()
        for idx, batch in tqdm(enumerate(inputs_tensor), total=len(inputs_tensor)):
            batch = batch.to(self.device)
            enc_out = self.model.encoder(inputs_embeds=batch)
            hidden_states = enc_out[0]
            encoder_outputs.append(hidden_states)

            result = self.model.generate(inputs=batch, max_length=1024, num_beams=num_beams, do_sample=False,
                                         length_penalty=0.4, eos_token_id=self.model.config.eos_token_id, early_stopping=False, bad_words_ids=None)
            result = self._postprocess_batch(result)
            single_event = self._to_event([result], [frame_times[idx]])

            # play something easy first, instrument existence
            instruments.update(set([note.program for note in single_event.notes if not note.is_drum]))
        
        return torch.cat(encoder_outputs, dim=0), instruments

    def _postprocess_batch(self, result):
        after_eos = torch.cumsum(
            (result == self.model.config.eos_token_id).float(), dim=-1)
        # minus special token
        result = result - self.vocab.num_special_tokens()
        result = torch.where(after_eos.bool(), -1, result)
        # remove bos
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result

    def _to_event(self, predictions_np: List[np.ndarray], frame_times: np.ndarray):
        predictions = []
        for i, batch in enumerate(predictions_np):
            for j, tokens in enumerate(batch):
                tokens = tokens[:np.argmax(
                    tokens == vocabularies.DECODED_EOS_ID)]
                start_time = frame_times[i][j][0]
                start_time -= start_time % (1 / self.codec.steps_per_second)
                # print('tokens', tokens[:10], 'start_time', start_time)
                predictions.append({
                    'est_tokens': tokens,
                    'start_time': start_time,
                    'raw_inputs': []
                })

        encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=encoding_spec)
        return result['est_ns']
