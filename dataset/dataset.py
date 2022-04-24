import json
from typing import List, Optional, Sequence, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import event_codec, note_sequences, spectrograms, vocabularies, run_length_encoding
from contrib.preprocessor import slakh_class_to_program_and_is_drum, add_track_to_notesequence, PitchBendError


class MidiMixDataset(Dataset):

    def __init__(self, root_dir) -> None:
        super().__init__()
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=1))
        self.is_train = True
        self.df = self._build_dataset(root_dir)
        self.include_ties = True
        self.ignore_pitch_bends = True
        self.onsets_only = False
        self.tie_token = self.codec.encode_event(event_codec.Event('tie', 0)) if self.include_ties else None

    def _build_dataset(self, root_dir):
        df = []
        audio_files = glob(f'{root_dir}/**/audio.wav')
        for a_f in audio_files:
            inst_path = a_f.replace('audio.wav', 'inst_names.json')
            with open(inst_path) as f:
                inst_names = json.load(f)
            midi_path = a_f.replace('audio.wav', 'midi')
            df.append({'inst_names': inst_names, 'audio_path': a_f, 'midi_path': midi_path})
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
    
    def _parse_midi(self, path, instrument_list):
        note_seqs = []
        for i, f in enumerate(instrument_list):
            note_seqs.append(note_seq.midi_file_to_note_sequence(f'{path}/{i}.mid'))
        return note_seqs

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
        for event in events:
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
                        output = np.concatenate(
                            [output, [output_steps]], axis=0)
                        shift_steps -= output_steps
                output = np.concatenate([output, [event]], axis=0)

        features[feature_key] = output
        return features

    def _compute_spectrogram(self, ex):
        samples = spectrograms.flatten_frames(ex['inputs'])
        ex['inputs'] = torch.from_numpy(np.array(spectrograms.compute_spectrogram(samples, self.spectrogram_config)))
        return ex

    def __getitem__(self, idx):
        row = self.df[idx]
        note_sequences = self._parse_midi(row['midi_path'], row['inst_names'])
        audio, sr = librosa.load(
            row['audio_path'], sr=None)
        if sr != self.spectrogram_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
        row = self._tokenize(note_sequences, audio, row['inst_names'])
        # TODO split to 2000 length -> to get more data
        # TODO select_random_chunk -> mel input length
        row = self._extract_target_sequence_with_indices(row, self.tie_token)
        row = self._run_length_encode_shifts(row)
        row = self._compute_spectrogram(row)
        return row

    def __len__(self):
        return len(self.df)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = MidiMixDataset(root_dir='/home/kunato/mt3/temp_data')
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    print(batch)
    print([(k, batch[k].shape) for k in batch.keys()])
    