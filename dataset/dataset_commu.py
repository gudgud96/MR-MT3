import torch
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import random
from typing import List, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import event_codec, note_sequences, spectrograms, vocabularies, run_length_encoding, metrics_utils
from contrib.preprocessor import add_track_to_notesequence, PitchBendError
import soundfile as sf
import os

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

# TODO: duplicated code...ideally, we should have a base class for all token-based transcription datasets

class ComMUDataset(Dataset):

    def __init__(
        self, 
        root_dir, 
        mel_length=256, 
        event_length=1024, 
        is_train=True, 
        include_ties=True, 
        ignore_pitch_bends=True, 
        onsets_only=False, 
        midi_folder='MIDI', 
        inst_filename='inst_names.json',
        shuffle=True,
        num_rows_per_batch=8
    ) -> None:
        super().__init__()
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=1))
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        self.midi_folder = midi_folder
        self.inst_filename = inst_filename
        self.mel_length = mel_length
        self.event_length = event_length
        self.df = self._build_dataset(root_dir, shuffle=shuffle)
        self.is_train = is_train
        self.include_ties = include_ties
        self.ignore_pitch_bends = ignore_pitch_bends
        self.onsets_only = onsets_only
        self.tie_token = self.codec.encode_event(event_codec.Event('tie', 0)) if self.include_ties else None
        self.num_rows_per_batch = num_rows_per_batch

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/*.wav'))
        print("root_dir", root_dir, len(audio_files))
        for a_f in audio_files:
            midi_path = a_f.replace("commu_audio_v2", "commu_midi_v2").replace("_16k.wav", ".mid")
            assert os.path.exists(midi_path)
            df.append({'audio_path': a_f, 'midi_path': midi_path})
        print('total file:', len(df))
        if shuffle:
            random.shuffle(df)
        return df

    def _audio_to_frames(
        self,
        samples: Sequence[float],
    ) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
        """Convert audio samples to non-overlapping frames and frame times."""
        frame_size = self.spectrogram_config.hop_width
        samples = np.pad(samples,
                         [0, frame_size - len(samples) % frame_size],
                         mode='constant')

        
        frames = spectrograms.split_audio(samples, self.spectrogram_config)

        num_frames = len(samples) // frame_size

        times = np.arange(num_frames) / \
            self.spectrogram_config.frames_per_second
        return frames, times

    def _parse_midi(self, path):
        return note_seq.midi_file_to_note_sequence(path)

    def _tokenize(self, track: note_seq.NoteSequence, samples: np.ndarray):

        frames, frame_times = self._audio_to_frames(samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)

        program, is_drum = track.notes[0].program, track.notes[0].is_drum
        try:
            add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                        ignore_pitch_bends=self.ignore_pitch_bends)
        except PitchBendError:
            # TODO(iansimon): is there a way to count these?
            return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        if self.is_train:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

        if self.onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(ns)
        else:
            times, values = (
                note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))
            # print('times', times)
            # print('values', values)

        (events, event_start_indices, event_end_indices,
         state_events, state_event_indices) = (
            run_length_encoding.encode_and_index_events(
                state=note_sequences.NoteEncodingState() if self.include_ties else None,
                event_times=times,
                event_values=values,
                encode_event_fn=note_sequences.note_event_data_to_events,
                codec=self.codec,
                frame_times=frame_times,
                encoding_state_to_events_fn=(
                    note_sequences.note_encoding_state_to_events
                    if self.include_ties else None)))

        # print('events', events)
        # print('inputs', np.array(frames).shape)
        # print('frame_times', frame_times.shape)
        # print('targets', events.shape)
        return {
            'inputs': np.array(frames),
            'input_times': frame_times,
            'targets': events,
            'input_event_start_indices': event_start_indices,
            'input_event_end_indices': event_end_indices,
            'state_events': state_events,
            'input_state_event_indices': state_event_indices,
            # 'sequence': ns.SerializeToString()
        }

    def _extract_target_sequence_with_indices(self, features, state_events_end_token=None):
        """Extract target sequence corresponding to audio token segment."""
        target_start_idx = features['input_event_start_indices'][0]
        target_end_idx = features['input_event_end_indices'][-1]

        features['targets'] = features['targets'][target_start_idx:target_end_idx]
        # print('features[targets]', [self.get_token_name(k) for k in features['targets']])

        if state_events_end_token is not None:
            # Extract the state events corresponding to the audio start token, and
            # prepend them to the targets array.
            state_event_start_idx = features['input_state_event_indices'][0]
            state_event_end_idx = state_event_start_idx + 1
            while features['state_events'][
                    state_event_end_idx - 1] != state_events_end_token:
                state_event_end_idx += 1
            features['targets'] = np.concatenate([
                features['state_events'][state_event_start_idx:state_event_end_idx],
                features['targets']
            ], axis=0)
            # print('features[targets] 2', [self.get_token_name(k) for k in features['targets']])

        return features

    def _run_length_encode_shifts(
        self,
        features,
        state_change_event_types=['velocity', 'program'],
        feature_key='targets'
    ):
        state_change_event_ranges = [self.codec.event_type_range(event_type)
                                     for event_type in state_change_event_types]

        events = features[feature_key]

        shift_steps = 0
        total_shift_steps = 0
        output = []

        current_state = np.zeros(
            len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            if self.codec.is_shift_event_index(event):
                shift_steps += 1
                total_shift_steps += 1
            else:
                # If this event is a state change and has the same value as the current
                # state, we can skip it entirely.

                # NOTE: this needs to be uncommented if not using random-order augmentation
                # because random-order augmentation use `_remove_redundant_tokens` to replace this part
                # is_redundant = False
                # for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                #     if (min_index <= event) and (event <= max_index):
                #         if current_state[i] == event:
                #             is_redundant = True
                #         current_state[i] = event
                # if is_redundant:
                #     continue

                # Once we've reached a non-shift event, RLE all previous shift events
                # before outputting the non-shift event.
                if shift_steps > 0:
                    shift_steps = total_shift_steps
                    while shift_steps > 0:
                        output_steps = np.minimum(
                            self.codec.max_shift_steps, shift_steps)
                        output = np.concatenate(
                            [output, [output_steps]], axis=0)
                        shift_steps -= output_steps
                output = np.concatenate([output, [event]], axis=0)

        features[feature_key] = output
        return features
    
    def _remove_redundant_tokens(
        self,
        events,
        state_change_event_types=['velocity', 'program'],
    ):
        state_change_event_ranges = [self.codec.event_type_range(event_type)
                                     for event_type in state_change_event_types]

        output = []

        current_state = np.zeros(
            len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            # If this event is a state change and has the same value as the current
            # state, we can skip it entirely.

            is_redundant = False
            for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                if (min_index <= event) and (event <= max_index):
                    if current_state[i] == event:
                        is_redundant = True
                    current_state[i] = event
            if is_redundant:
                continue

            # Once we've reached a non-shift event, RLE all previous shift events
            # before outputting the non-shift event.
            output = np.concatenate([output, [event]], axis=0)

        return output

    def _compute_spectrogram(self, ex):
        samples = spectrograms.flatten_frames(ex['inputs'])
        ex['inputs'] = torch.from_numpy(np.array(spectrograms.compute_spectrogram(samples, self.spectrogram_config)))
        # add normalization
        ex['inputs'] = torch.clamp(ex['inputs'], min=MIN_LOG_MEL, max=MAX_LOG_MEL)
        ex['inputs'] = (ex['inputs'] - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
        return ex

    def _pad_length(self, row):
        inputs = row['inputs'][:self.mel_length].to(torch.float32)
        targets = torch.from_numpy(row['targets'][:self.event_length]).to(torch.long)
        targets = targets + self.vocab.num_special_tokens()
        if inputs.shape[0] < self.mel_length:
            pad = torch.zeros(self.mel_length - inputs.shape[0], inputs.shape[1], dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat([inputs, pad], dim=0)
        # print('targets shape', targets.shape[0])
        if targets.shape[0] < self.event_length:
            eos = torch.ones(1, dtype=targets.dtype, device=targets.device)
            if self.event_length - targets.shape[0] - 1 > 0:
                pad = torch.ones(self.event_length - targets.shape[0] - 1, dtype=targets.dtype, device=targets.device) * -100
                targets = torch.cat([targets, eos, pad], dim=0)
            else:
                targets = torch.cat([targets, eos], dim=0)
        return {'inputs': inputs, 'targets': targets, "input_times": row["input_times"]}
    
    def _split_frame(self, row, length=2000):
        rows = []
        input_length = row['inputs'].shape[0]

        # chunk the audio into chunks of `length` = 2000
        # during _random_chunk, within each chunk, randomly select a segment = self.mel_length
        for split in range(0, input_length, length):
            if split + length >= input_length:
                continue
            new_row = {}
            for k in row.keys():
                if k in ['inputs', 'input_times', 'input_event_start_indices', 'input_event_end_indices', 'input_state_event_indices']:
                    new_row[k] = row[k][split:split+length]
                else:
                    new_row[k] = row[k]
            rows.append(new_row)
        
        if len(rows) == 0:
            return [row]
        return rows
    
    def _random_chunk(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        start_length = random.randint(0, random_length)
        for k in row.keys():
            if k in ['inputs', 'input_times', 'input_event_start_indices', 'input_event_end_indices', 'input_state_event_indices']:
                new_row[k] = row[k][start_length:start_length+self.mel_length]
            else:
                new_row[k] = row[k]
        return new_row
    
    def _to_event(self, predictions_np: List[np.ndarray], frame_times: np.ndarray):
        predictions = []
        for i, batch in enumerate(predictions_np):
            for j, tokens in enumerate(batch):
                # tokens = tokens[:np.argmax(
                #     tokens == vocabularies.DECODED_EOS_ID)]
                start_time = frame_times[i][j][0]
                start_time -= start_time % (1 / self.codec.steps_per_second)
                predictions.append({
                    'est_tokens': tokens,
                    'start_time': start_time,
                    'raw_inputs': []
                })

        encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=encoding_spec)
        return result['est_ns']
    
    def _postprocess_batch(self, result):
        EOS_TOKEN_ID = 1    # TODO: this is a hack!
        after_eos = torch.cumsum(
            (result == EOS_TOKEN_ID).float(), dim=-1)
        # minus special token
        result = result - self.vocab.num_special_tokens()
        result = torch.where(after_eos.bool(), -1, result)
        # remove bos
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result
    
    def __getitem__(self, idx):
        row = self.df[idx]
        ns = self._parse_midi(row['midi_path'])
        audio, sr = librosa.load(row['audio_path'], sr=None)
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        row = self._tokenize(ns, audio)

        # NOTE: by default, this is self._split_frame(row, length=2000)
        # this does not guarantee the chunks in `rows` to be contiguous.
        # if we need to ensure that the chunks in `rows` to be contiguous, use:
        rows = self._split_frame(row, length=self.mel_length)
        
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

            # # print("start_times", row["input_times"][0])
            # if fake_start is None:
            #     fake_start = row["input_times"][0]
            # # predictions = []
            # predictions.append({
            #     'est_tokens': result.cpu().detach().numpy(),    # has to be numpy here, or else problematic
            #     'start_time': row["input_times"][0] - fake_start,
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
    
    def __len__(self):
        return len(self.df)
    
    def randomize_tokens(self, token_lst):
        shift_idx = [i for i in range(len(token_lst)) if "shift" in token_lst[i]]
        if len(shift_idx) == 0:
            return token_lst
        res = token_lst[:shift_idx[0]]
        for j in range(len(shift_idx) - 1):
            res += [token_lst[shift_idx[j]]]
            
            start_idx = shift_idx[j]
            end_idx = shift_idx[j + 1]
            cur = token_lst[start_idx + 1 : end_idx]
            cur_lst = []
            ptr = 0
            while ptr < len(cur):
                t = cur[ptr]
                if "program" in t:
                    cur_lst.append([cur[ptr], cur[ptr + 1], cur[ptr + 2]])
                    ptr += 3
                elif "velocity" in t:
                    cur_lst.append([cur[ptr], cur[ptr + 1]])
                    ptr += 2

            indices = np.arange(len(cur_lst))
            np.random.shuffle(indices)

            new_cur_lst = []
            for idx in indices:
                new_cur_lst.append(cur_lst[idx])
            
            new_cur_lst = [item for sublist in new_cur_lst for item in sublist]
            res += new_cur_lst
        
        res += token_lst[shift_idx[-1]:]
        return res
    
    def get_token_name(self, token_idx):
        token_idx = int(token_idx)
        if token_idx >= 1001 and token_idx <= 1128:
            token = f"pitch_{token_idx - 1001}"
        elif token_idx >= 1129 and token_idx <= 1130:
            token = f"velocity_{token_idx - 1129}"
        elif token_idx >= 1131 and token_idx <= 1131:
            token = "tie"
        elif token_idx >= 1132 and token_idx <= 1259:
            token = f"program_{token_idx - 1132}"
        elif token_idx >= 1260 and token_idx <= 1387:
            token = f"drum_{token_idx - 1260}"
        elif token_idx >= 0 and token_idx < 1000:
            token = f"shift_{token_idx}"
        else:
            token = f"invalid_{token_idx}"
        
        return token
    
    def token_to_idx(self, token_str):
        if "pitch" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1001
        elif "velocity" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1129
        elif "tie" in token_str:
            token_idx = 1131
        elif "program" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1132
        elif "drum" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1260
        elif "shift" in token_str:
            token_idx = int(token_str.split("_")[1])

        return token_idx


def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    return inputs, targets

if __name__ == '__main__':
    dataset = ComMUDataset(
        root_dir='/data/datasets/ComMU/dataset_processed/commu_audio_v2/train/',
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256
    )
    print("pitch", dataset.codec.event_type_range("pitch"))
    print("velocity", dataset.codec.event_type_range("velocity"))
    print("tie", dataset.codec.event_type_range("tie"))
    print("program", dataset.codec.event_type_range("program"))
    print("drum", dataset.codec.event_type_range("drum"))
    dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=True)
    for idx, item in enumerate(dl):
        inputs, targets = item
        print(idx, inputs.shape, targets[0].shape)
        break
    