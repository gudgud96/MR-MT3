import torch
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import event_codec, note_sequences, spectrograms, vocabularies, run_length_encoding, metrics_utils
from contrib.preprocessor import slakh_class_to_program_and_is_drum, add_track_to_notesequence, PitchBendError
import soundfile as sf

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

class SlakhDataset(Dataset):

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
        self.audio_filename = audio_filename
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
        self.drum_program = 128 # 0-127 are instruments

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/**/{self.audio_filename}'))
        print("root_dir", root_dir, len(audio_files), self.audio_filename)
        for a_f in audio_files:
            inst_path = a_f.replace(self.audio_filename, self.inst_filename)
            midi_path = a_f.replace(self.audio_filename, self.midi_folder)
            with open(inst_path) as f:
                inst_names = json.load(f)
            df.append({'inst_names': inst_names, 'audio_path': a_f, 'midi_path': midi_path})
        assert len(df) > 0
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

    def _parse_midi(self, path, instrument_dict: Dict[str, str]):
        note_seqs = []

        for filename in instrument_dict.keys():
            # this can be pretty_midi.PrettyMIDI() obj / string path to midi
            midi_path = f'{path}/{filename}.mid'
            note_seqs.append(note_seq.midi_file_to_note_sequence(midi_path))
        return note_seqs, instrument_dict.values()

    def _tokenize(self, tracks: List[note_seq.NoteSequence], samples: np.ndarray, inst_names: List[str], example_id: Optional[str] = None):

        frames, frame_times = self._audio_to_frames(samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        # tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        assert len(tracks) == len(inst_names)

        for track, inst_name in zip(tracks, inst_names):
            # Instrument name should be Slakh class.
            program, is_drum = slakh_class_to_program_and_is_drum(inst_name)
            try:
                add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                          ignore_pitch_bends=self.ignore_pitch_bends)
            except PitchBendError:
                # TODO(iansimon): is there a way to count these?
                return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        # if self.is_train:
        #     # Trim overlapping notes in training (as our event vocabulary cannot
        #     # represent them), but preserve original NoteSequence for eval.
        #     ns = note_sequences.trim_overlapping_notes(ns)

        if example_id is not None:
            ns.id = example_id

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
    
    def _tokenize_new(self, tracks: List[note_seq.NoteSequence], samples: np.ndarray, inst_names: List[str], example_id: Optional[str] = None):

        frames, frame_times = self._audio_to_frames(samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        # tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        assert len(tracks) == len(inst_names)

        for track, inst_name in zip(tracks, inst_names):
            # Instrument name should be Slakh class.
            program, is_drum = slakh_class_to_program_and_is_drum(inst_name)
            try:
                add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                          ignore_pitch_bends=self.ignore_pitch_bends)
            except PitchBendError:
                # TODO(iansimon): is there a way to count these?
                return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        # if self.is_train:
        #     # Trim overlapping notes in training (as our event vocabulary cannot
        #     # represent them), but preserve original NoteSequence for eval.
        #     ns = note_sequences.trim_overlapping_notes(ns)

        # make target_seq a numpy
        # use program = 128 to represent drums
        target_seq = np.array([
            [i.pitch,
            i.program if i.is_drum==False else self.drum_program,
            i.start_time,
            i.end_time] for i in ns.notes])

        # print('events', events)
        # print('inputs', np.array(frames).shape)
        # print('frame_times', frame_times.shape)
        # print('targets', events.shape)
        return {
            'inputs': np.array(frames),
            'input_times': frame_times,
            'targets': target_seq,
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
        if targets.shape[0] < self.event_length:
            eos = torch.ones(1, dtype=targets.dtype, device=targets.device)
            if self.event_length - targets.shape[0] - 1 > 0:
                pad = torch.ones(self.event_length - targets.shape[0] - 1, dtype=targets.dtype, device=targets.device) * -100
                targets = torch.cat([targets, eos, pad], dim=0)
            else:
                targets = torch.cat([targets, eos], dim=0)
        return {'inputs': inputs, 'targets': targets, "input_times": row["input_times"]}
    
    def _pad_length_targets(self, row):
        # TODO: raise all classes by 1 to inclue Empty token
        assert row['targets'].shape[0] <= self.event_length, f"Number of events ({row['targets'].shape[0]}) exceeds the max_event ({self.event_length})"
        for k in row.keys():
            if k in ['targets', 'global_frames', 'local_frames'] and row[k].shape[0] < self.event_length:
                # add a dimension to the numpy array if it is 1D
                if row[k].ndim == 1:
                    # expect the numpy array to be 2D
                    row[k] = row[k].reshape(-1, 1)
                pad = np.ones((self.event_length - row[k].shape[0], row[k].shape[1]), dtype=row[k].dtype) * -100
                row[k] = np.concatenate([row[k], pad], axis=0)
        return row
    
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
    
    def _split_frame_new(self, row, length=2000):
        splits = []
        input_length = row['inputs'].shape[0]

        # convert time in seconds into time in frames
        start_frame = np.round(row['targets'][:, 2] * self.spectrogram_config.frames_per_second)
        end_frame = np.round(row['targets'][:, 3] * self.spectrogram_config.frames_per_second)
        # storing two extra info for debugging
        global_frames = []
        local_frames = []        

        # chunk the audio into splits of `length` = 2000
        # during _random_chunk, within each chunk, randomly select a segment = self.mel_length
        for split in range(0, input_length, length):
            new_row = {}
            if split + length >= input_length: # discard the last chunk if it is too short
                continue

            start_idx = int(split)
            end_idx = int(split + length)
            # creating a mask to find active events within the chunk
            new_row['inputs'] = row['inputs'][start_idx:end_idx]
            active_mask = np.bitwise_and((start_frame < end_idx), (end_frame > start_idx))
            new_row['global_frames'] = np.stack((start_frame, end_frame), axis=1)[active_mask]
            new_row['local_frames'] = np.stack((start_frame - split, end_frame - split), axis=1)[active_mask]
            new_row['targets'] = row['targets'][active_mask]
            splits.append(new_row)
        
        return splits
    
    def _random_chunk(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        # start_length = random.randint(0, random_length)
        # TODO: revert back to normal after debugging
        start_length = 0
        for k in row.keys():
            if k in ['inputs', 'input_times', 'input_event_start_indices', 'input_event_end_indices', 'input_state_event_indices']:
                new_row[k] = row[k][start_length:start_length+self.mel_length]
            else:
                new_row[k] = row[k]
        return new_row
    
    def _random_chunk_new(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        start_length = random.randint(0, random_length)     

        start_idx = start_length
        end_idx = start_length + self.mel_length            
        active_mask = np.bitwise_and((row['local_frames'][:,0] < end_idx), (row['local_frames'][:,1] > start_idx))

        new_row['targets'] = torch.from_numpy(row['targets'][active_mask])
        new_row['global_frames'] = torch.from_numpy(row['global_frames'][active_mask])
        new_row['local_frames'] = torch.from_numpy(row['local_frames'][active_mask]) - start_length
        new_row['inputs'] = row['inputs'][start_length:start_length+self.mel_length]

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
        ns, inst_names = self._parse_midi(row['midi_path'], row['inst_names'])
        audio, sr = librosa.load(row['audio_path'], sr=None)
        filename = row['audio_path'].split('/')[-2]
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        row_new = self._tokenize_new(ns, audio, inst_names) # convert noteseq data to numpy array
        # splitting the full audio/token sequence into chunks of 2000 frames
        splits = self._split_frame_new(row_new)
        
        wavs, inputs, targets, debug_targets, debug_local_frames = [], [], [], [], []

        # forming a batch based on self.num_rows_per_batch
        if len(splits) > self.num_rows_per_batch:
            start_idx = random.randint(0, len(splits) - self.num_rows_per_batch)
            splits = splits[start_idx : start_idx + self.num_rows_per_batch]

        for j, row_new in enumerate(splits):
            row_new = self._random_chunk_new(row_new)

            ## uncomment this to debug the tokens with audio
            # wavs.append(row_new["inputs"].reshape(-1,))

            row_new = self._compute_spectrogram(row_new)
            # row_new = self._pad_length_targets(row_new) # new this to create negative samples
            ## uncomment this to debug the tokens with audio
            # for i, event in enumerate(row_new["targets"]):
            #     print(f"{event[2]} <time_{row_new['local_frames'][i][0]}, program_{event[1]}, pitch_{event[0]}>")
            # debug_local_frames.append(row_new['local_frames'] +  j*256) # one chunk = 256 frames
            # debug_targets.append(row_new['targets'])

            # target_dict = {}

            # considering instrument as the class
            # pitch, onset, offset as the bounding box
            # target_dict['labels'] = row_new["targets"][:,1]
            # target_dict['boxes'] =  np.concatenate(
            #     (row_new["targets"][:,:1], # obtain pitch but keep the dim
            #     row_new['local_frames']), axis=1) # add back time in frames

            target_in_frames = torch.cat(
                    (row_new["targets"][:,:2], # remove time in seconds
                     row_new['local_frames']), dim=1) # add back time in frames
            
            targets.append(target_in_frames.long())
            inputs.append(row_new["inputs"])

        ## ================= exporting MIDI and audio ====================
        ## using absolute time info
        # complete_sequence = np.vstack(i for i in debug_targets)
        # # we can remove duplicated notes by using np.unique
        # complete_sequence_unique = np.unique(complete_sequence, axis=0)
        # print(f"Number of duplicated events removed: {len(complete_sequence_unique) - len(complete_sequence)}")        

        # note_output = note_seq.NoteSequence(ticks_per_quarter=220)
        # for event in complete_sequence_unique:
        #     if event[0] !=-100: # ignore padding
        #         note_output.notes.add(
        #             pitch=int(event[0]),
        #             start_time=event[2],
        #             end_time=event[3],
        #             velocity=80,
        #             program=int(event[1]) if event[1]!=129 else 0,
        #             is_drum=True if event[1]==129 else False,
        #             ) 
            
        # note_sequences.assign_instruments(note_output)
        # note_sequences.validate_note_sequence(note_output)
        # # everything seems okay
        # note_seq.note_sequence_to_midi_file(note_output, 'decode_realtime.mid')

        # # reconstruct using local frames
        # complete_sequence = np.vstack(
        #     np.concatenate(
        #         (i[:,:2],
        #          j[:]/self.spectrogram_config.frames_per_second), axis=1) for i,j in zip(debug_targets, debug_local_frames))        
        # complete_sequence_unique = np.unique(complete_sequence, axis=0)
        # note_output = note_seq.NoteSequence(ticks_per_quarter=220)
        # for event in complete_sequence_unique:
        #     if event[0] !=-100: # ignore padding
        #         note_output.notes.add(
        #             pitch=int(event[0]),
        #             start_time=event[2],
        #             end_time=event[3] if event[1]!=129 else event[2] + 0.01,
        #             velocity=80,
        #             program=int(event[1]) if event[1]!=129 else 0,
        #             is_drum=True if event[1]==129 else False,
        #             ) 
            
        # note_sequences.assign_instruments(note_output)
        # note_sequences.validate_note_sequence(note_output)
        # # everything seems okay
        # note_seq.note_sequence_to_midi_file(note_output, 'decode_frametime.mid')   
        # sf.write(f"test_out.wav", np.concatenate(wavs), 16000, "PCM_24")


        return torch.stack(inputs), targets
    
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
    

class SlakhDatasetDuration(SlakhDataset):

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
        midi_folder='MIDI', 
        inst_filename='inst_names.json',
        shuffle=True,
        num_rows_per_batch=8
    ) -> None:
        # pass all arguments to parent class
        super().__init__(root_dir,
                         mel_length,
                         event_length,
                         is_train,
                         include_ties,
                         ignore_pitch_bends,
                         onsets_only,
                         audio_filename,
                         midi_folder,
                         inst_filename,
                         shuffle,
                         num_rows_per_batch)
 
    def _split_frame_new(self, row, length=2000):
        splits = []
        input_length = row['inputs'].shape[0]

        # convert time in seconds into time in frames
        start_frame = np.round(row['targets'][:, 2] * self.spectrogram_config.frames_per_second)
        end_frame = np.round(row['targets'][:, 3] * self.spectrogram_config.frames_per_second)   

        # chunk the audio into splits of `length` = 2000
        # during _random_chunk, within each chunk, randomly select a segment = self.mel_length
        for split in range(0, input_length, length):
            new_row = {}
            if split + length >= input_length: # discard the last chunk if it is too short
                continue

            start_idx = int(split)
            end_idx = int(split + length)
            # creating a mask to find active events within the chunk
            new_row['inputs'] = row['inputs'][start_idx:end_idx]
            active_mask = np.bitwise_and((start_frame < end_idx), (end_frame > start_idx))
            new_row['global_frames'] = np.stack((start_frame, end_frame), axis=1)[active_mask]
            local_start_frame = start_frame - split
            local_end_frame = end_frame - split
            duration = local_end_frame.copy() # initialize the duration for the np.subtract function
            new_row['local_frames'] = np.stack((local_start_frame, local_end_frame), axis=1)[active_mask]
            new_row['duration'] = np.subtract(local_end_frame,
                                              local_start_frame,
                                              where=local_start_frame>=0,
                                              out=duration)[active_mask]
            new_row['targets'] = row['targets'][active_mask]
            splits.append(new_row)
        
        return splits
    
    def _random_chunk_new(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        start_length = random.randint(0, random_length)     

        start_idx = start_length
        end_idx = start_length + self.mel_length            
        active_mask = np.bitwise_and((row['local_frames'][:,0] < end_idx), (row['local_frames'][:,1] > start_idx))

        new_row['targets'] = torch.from_numpy(row['targets'][active_mask])
        new_row['global_frames'] = torch.from_numpy(row['global_frames'][active_mask])
        new_row['local_frames'] = torch.from_numpy(row['local_frames'][active_mask]) - start_length
        new_row['duration'] = torch.from_numpy(row['duration'][active_mask])
        new_row['inputs'] = row['inputs'][start_length:start_length+self.mel_length]

        return new_row
    
    
    def __getitem__(self, idx):
        row = self.df[idx]
        ns, inst_names = self._parse_midi(row['midi_path'], row['inst_names'])
        audio, sr = librosa.load(row['audio_path'], sr=None)
        filename = row['audio_path'].split('/')[-2]
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        row_new = self._tokenize_new(ns, audio, inst_names) # convert noteseq data to numpy array
        # splitting the full audio/token sequence into chunks of 2000 frames
        splits = self._split_frame_new(row_new)
        
        wavs, inputs, targets, debug_targets, debug_local_frames = [], [], [], [], []

        # forming a batch based on self.num_rows_per_batch
        if len(splits) > self.num_rows_per_batch:
            start_idx = random.randint(0, len(splits) - self.num_rows_per_batch)
            splits = splits[start_idx : start_idx + self.num_rows_per_batch]

        for j, row_new in enumerate(splits):
            row_new = self._random_chunk_new(row_new)
            row_new = self._compute_spectrogram(row_new)

            target_in_frames = torch.cat(
                    (row_new["targets"][:,:2], # remove time in seconds
                     row_new['local_frames'][:], # add onset
                     row_new['duration'].unsqueeze(-1) # add duration
                     ), dim=1) 
            
            inputs.append(row_new["inputs"])
            targets.append(target_in_frames.long())
            # target will be a tensor of shape (num_events, 5)
            # The 5 columns are: pitch, program, onset, offset, duration

        return torch.stack(inputs), targets
    
def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    flatten_targets = []

    for k in lst:
        flatten_targets.extend(k[1])
    # num_insts = torch.cat([k[2] for k in lst])

    # add random shuffling here
    # indices = np.arange(inputs.shape[0])
    # np.random.shuffle(indices)
    # indices = torch.from_numpy(indices)
    # inputs = inputs[indices]
    # targets = targets[indices]
    # num_insts = num_insts[indices]

    return inputs, flatten_targets

if __name__ == '__main__':
    dataset = SlakhDataset(
        root_dir='/data/slakh2100_flac_redux/test/',
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
    dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
    for idx, item in enumerate(dl):
        inputs, targets = item
        print(idx, inputs.shape, targets[0])
        break
    


def pred2midi2(events, output_filename='new_token_decoded.mid', verbose=False):
    step2time_factor = 125

    state = None
    prev_event = [-1, -1, -1, -1] # non active event
    
    # state = 1 if note is on
    # state = 0 if note is off
    ophand_events = [] # a list to store onset events without offsets

    # putting everything back to ns_seq
    # for drums it is instrument 9??
    note_output = note_seq.NoteSequence(ticks_per_quarter=220)
    for idx, event in enumerate(events):
        message = []
        event = event.tolist()
        state = event[2]        
        if event[1]==128: # when the current event is a drum (program 128)
            note_output.notes.add(
                pitch=int(event[0]),
                start_time= event[3] / step2time_factor,
                end_time=(event[3] / step2time_factor)+1, # dummy offset for drums
                velocity=120,
                program=0,
                is_drum=True,
                instrument=0
                )
            # the drum event is comppleted, no need to save the event        
        elif state == 1 and prev_event[2]!=state and event[1]!=self.drum_program: # when the current event is an onset
            onset_time = event[3] / step2time_factor # TODO: don't use hard-coded value
            prev_event = event # use this to look for the offset event
        elif state == 0 and prev_event[2]!=state: # when the current event is an offset
            if prev_event[0] == event[0]: # check if the pitch is the same as the previous onset event
                note_output.notes.add(
                    pitch=int(event[0]),
                    start_time=onset_time,
                    end_time=event[3] / step2time_factor,
                    velocity=80,
                    program=int(event[1]),
                    is_drum=False,
                    instrument=0
                    )
                if prev_event[2] == state:
                    message.append(f'Error: two consecutive off events @ {idx}')
                    ophand_events.append(event) # temporary put the event as ophand event
                else:
                    prev_event = event # use this to look for the onset event
            else: # if the pitch is different, and put current and previous event to the ophand event list
                # raise warning
                message.append(f'The offset event has a different pitch than the onset event @ {idx}')
                ophand_events.append(event)
                ophand_events.append(prev_event)
                # reset prev_event
                prev_event = [-1, -1, -1, -1]
        else:
            message.append(f'Error: two consecutive events @ {idx}\n{event=}, {prev_event=}')
            ophand_events.append(event)
            ophand_events.append(prev_event)
            # reset prev_event
            prev_event = [-1, -1, -1, -1]


    if verbose:
        print(message)

    note_sequences.assign_instruments(note_output)
    note_sequences.validate_note_sequence(note_output)
    note_seq.note_sequence_to_midi_file(note_output, output_filename)
    return note_output, ophand_events