import torch
from torch.utils.data import DataLoader

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import random
import glob
import json
from typing import Dict, List, Optional, Sequence, Tuple

from dataset_2_random import SlakhDataset
import librosa
import numpy as np
import note_seq

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

class SlakhStemMixDataset(SlakhDataset):

    def __init__(
        self, 
        root_dir, 
        mel_length=256, 
        event_length=1024, 
        is_train=True, 
        include_ties=True, 
        ignore_pitch_bends=True, 
        onsets_only=False, 
        audio_filename='mix.flac', 
        stems_folder='stems',
        midi_folder='MIDI', 
        inst_filename='inst_names.json',
        shuffle=True,
        num_rows_per_batch=8
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            mel_length=mel_length,
            event_length=event_length,
            is_train=is_train,
            include_ties=include_ties,
            ignore_pitch_bends=ignore_pitch_bends,
            onsets_only=onsets_only,
            audio_filename=audio_filename,
            midi_folder=midi_folder,
            inst_filename=inst_filename,
            shuffle=shuffle,
            num_rows_per_batch=num_rows_per_batch
        )
        self.stems_folder = stems_folder

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/**/{self.audio_filename}'))
        print("root_dir", root_dir, len(audio_files), self.audio_filename)
        for a_f in audio_files:
            # get path for stems
            stems_path = a_f.replace(self.audio_filename, self.stems_folder)
            stem_audio_files = sorted(glob(f'{stems_path}/*.flac'))
            stem_midi_files = [k.replace(self.stems_folder, self.midi_folder).replace(".flac", ".mid") for k in stem_audio_files]


            inst_path = a_f.replace(self.audio_filename, self.inst_filename)
            with open(inst_path) as f:
                inst_names = json.load(f)
            print({'inst_names': inst_names, 'audio_path': stem_audio_files, 'midi_path': stem_midi_files})
            df.append({'inst_names': inst_names, 'audio_path': stem_audio_files, 'midi_path': stem_midi_files})
        
        assert len(df) > 0
        print('total file:', len(df))
        if shuffle:
            random.shuffle(df)
        return df

    def _parse_midi(self, path, instrument_dict: Dict[str, str]):
        note_seqs = []

        for filename in instrument_dict.keys():
            # this can be pretty_midi.PrettyMIDI() obj / string path to midi
            midi_path = f'{path}/{filename}.mid'
            note_seqs.append(note_seq.midi_file_to_note_sequence(midi_path))
        return note_seqs, instrument_dict.values()

    def __getitem__(self, idx):
        row = self.df[idx]
        ns, inst_names = self._parse_midi(row['midi_path'], row['inst_names'])
        audio, sr = librosa.load(row['audio_path'], sr=None)
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        
        # TODO: ns and audio needs to be augmented from randomly selected stems
        
        row = self._tokenize(ns, audio, inst_names)

        # NOTE: by default, this is self._split_frame(row, length=2000)
        # this does not guarantee the chunks in `rows` to be contiguous.
        # if we need to ensure that the chunks in `rows` to be contiguous, use:
        # rows = self._split_frame(row, length=self.mel_length)
        rows = self._split_frame(row)
        
        inputs, targets, frame_times, num_insts = [], [], [], []
        if len(rows) > self.num_rows_per_batch:
            start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx : start_idx + self.num_rows_per_batch]
        
        predictions = []
        # wavs = []
        fake_start = None
        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            row = self._run_length_encode_shifts(row)
            
            # wavs.append(row["inputs"].reshape(-1,))
            # sf.write(f"test_{j}.wav", row["inputs"].reshape(-1,), 16000, "PCM_24")

            row = self._compute_spectrogram(row)

            # -- random order augmentation --
            # If turned on, comment out `is_redundant` code in `run_length_encoding`
            # print("=======")
            # print(j, [self.get_token_name(t) for t in row["targets"]])
            t = self.randomize_tokens([self.get_token_name(t) for t in row["targets"]])
            t = np.array([self.token_to_idx(k) for k in t])
            t = self._remove_redundant_tokens(t)
            row["targets"] = t
            
            row = self._pad_length(row)
            inputs.append(row["inputs"])
            targets.append(row["targets"])   

            # ========== for reconstructing the MIDI from events =========== #
            # result = row["targets"]
            # EOS_TOKEN_ID = 1    # TODO: this is a hack!
            # after_eos = torch.cumsum(
            #     (result == EOS_TOKEN_ID).float(), dim=-1
            # )
            # result -= self.vocab.num_special_tokens()
            # result = torch.where(after_eos.bool(), -1, result)

            # print("start_times", row["input_times"][0])
            # if fake_start is None:
            #     fake_start = row["input_times"][0]
            # # predictions = []
            # predictions.append({
            #     'est_tokens': result.cpu().detach().numpy(),    # has to be numpy here, or else problematic
            #     # 'start_time': row["input_times"][0] - fake_start,
            #     'start_time': j * 2.048,
            #     # 'start_time': 0,
            #     'raw_inputs': []
            # })

            # # encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            # # result = metrics_utils.event_predictions_to_ns(
            # #     predictions, codec=self.codec, encoding_spec=encoding_spec)
            # # note_seq.sequence_proto_to_midi_file(result['est_ns'], f"test_out_{j}.mid")   
            # sf.write(f"test_out.wav", np.concatenate(wavs), 16000, "PCM_24")

             # ========== for reconstructing the MIDI from events =========== #
        
        # ========== for reconstructing the MIDI from events =========== #
        # encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        # result = metrics_utils.event_predictions_to_ns(
        #     predictions, codec=self.codec, encoding_spec=encoding_spec)
        # note_seq.sequence_proto_to_midi_file(result['est_ns'], "test_out.mid")   
        # sf.write(f"test_out.wav", np.concatenate(wavs), 16000, "PCM_24")
        # ========== for reconstructing the MIDI from events =========== #  
        # num_insts = np.stack(num_insts)

        return torch.stack(inputs), torch.stack(targets)
    

def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    # num_insts = torch.cat([k[2] for k in lst])

    # add random shuffling here
    # indices = np.arange(inputs.shape[0])
    # np.random.shuffle(indices)
    # indices = torch.from_numpy(indices)
    # inputs = inputs[indices]
    # targets = targets[indices]
    # num_insts = num_insts[indices]

    return inputs, targets

if __name__ == '__main__':
    dataset = SlakhStemMixDataset(
        root_dir='/data/slakh2100_flac_redux/test/',
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256
    )
    # print("pitch", dataset.codec.event_type_range("pitch"))
    # print("velocity", dataset.codec.event_type_range("velocity"))
    # print("tie", dataset.codec.event_type_range("tie"))
    # print("program", dataset.codec.event_type_range("program"))
    # print("drum", dataset.codec.event_type_range("drum"))
    # dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
    # for idx, item in enumerate(dl):
    #     inputs, targets = item
    #     print(idx, inputs.shape, targets[0])
    #     break
    