import torch
from torch.utils.data import Dataset, DataLoader

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

from itertools import cycle
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import event_codec, note_sequences, spectrograms, vocabularies, run_length_encoding, metrics_utils
from contrib.preprocessor import slakh_class_to_program_and_is_drum, add_track_to_notesequence, PitchBendError


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
        shuffle=True
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

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/**/{self.audio_filename}'))
        print("root_dir", root_dir, len(audio_files))
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
        if self.is_train:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

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

    def _extract_target_sequence_with_indices(self, features, state_events_end_token=None):
        """Extract target sequence corresponding to audio token segment."""
        target_start_idx = features['input_event_start_indices'][0]
        target_end_idx = features['input_event_end_indices'][-1]

        features['targets'] = features['targets'][target_start_idx:target_end_idx]

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
            # if j <= 100:
            #     print(j, event, self.codec.decode_event_index(event))
            if self.codec.is_shift_event_index(event):
                shift_steps += 1
                total_shift_steps += 1
            else:
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
                if shift_steps > 0:
                    shift_steps = total_shift_steps
                    while shift_steps > 0:
                        output_steps = np.minimum(
                            self.codec.max_shift_steps, shift_steps)
                        # print("output_steps", output_steps)
                        output = np.concatenate(
                            [output, [output_steps]], axis=0)
                        shift_steps -= output_steps
                    # print("shift_steps", shift_steps, total_shift_steps)
                output = np.concatenate([output, [event]], axis=0)

        features[feature_key] = output
        return features

    def _compute_spectrogram(self, ex):
        samples = spectrograms.flatten_frames(ex['inputs'])
        ex['inputs'] = torch.from_numpy(np.array(spectrograms.compute_spectrogram(samples, self.spectrogram_config)))
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
    
    def _split_frame(self, row, length=2000):
        rows = []
        input_length = row['inputs'].shape[0]
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
        """
        TODO: Should be good enough for training now. 
        Although there is still one imperfection for reconstruction, 
        which is the `start_time` for each segment seems to be wrong, hence holding the track back.
        """
        row = self.df[idx]
        ns, inst_names = self._parse_midi(row['midi_path'], row['inst_names'])
        audio, sr = librosa.load(row['audio_path'], sr=None)
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        row = self._tokenize(ns, audio, inst_names)
        rows = self._split_frame(row)
        
        inputs, targets, frame_times = [], [], []
        num_rows = 8
        if len(rows) > num_rows:
            start_idx = random.randint(0, len(rows) - num_rows)
            # start_idx = 0
            rows = rows[start_idx : start_idx + num_rows]
        
        predictions = []
        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            row = self._run_length_encode_shifts(row)
            row = self._compute_spectrogram(row)
            row = self._pad_length(row)
            inputs.append(row["inputs"])
            targets.append(row["targets"])   

            # ========== for reconstructing the MIDI from events =========== #
            # print(j)
            # print('est_tokens', row["targets"].shape)
            
            # result = row["targets"]
            # EOS_TOKEN_ID = 1    # TODO: this is a hack!
            # after_eos = torch.cumsum(
            #     (result == EOS_TOKEN_ID).float(), dim=-1
            # )
            # result -= self.vocab.num_special_tokens()
            # result = torch.where(after_eos.bool(), -1, result)

            # print("start_times", row["input_times"][0])
            # predictions.append({
            #     'est_tokens': result.cpu().detach().numpy(),    # has to be numpy here, or else problematic
            #     'start_time': row["input_times"][0],
            #     'raw_inputs': []
            # })
             # ========== for reconstructing the MIDI from events =========== #
        
        # ========== for reconstructing the MIDI from events =========== #
        # encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        # result = metrics_utils.event_predictions_to_ns(
        #     predictions, codec=self.codec, encoding_spec=encoding_spec)
        # note_seq.sequence_proto_to_midi_file(result['est_ns'], "test_out.mid")   
        # ========== for reconstructing the MIDI from events =========== #  
    
        return torch.stack(inputs), torch.stack(targets)
    
    def __len__(self):
        return len(self.df)


def collate_fn(lst):
    inputs = [k[0] for k in lst]
    targets = [k[1] for k in lst]
    return torch.cat(inputs), torch.cat(targets)

if __name__ == '__main__':
    dataset = SlakhDataset(
        root_dir='/data/slakh2100_flac_redux/test/',
        shuffle=False,
        is_train=False,
        include_ties=True
    )
    dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
    for idx, item in enumerate(dl):
        inputs, targets = item
        print(idx, inputs.shape, targets.shape)
        break
    