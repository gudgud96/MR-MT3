import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from dataset.dataset_2_random import SlakhDataset

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

class SlakhDatasetWithPrevSegmem(SlakhDataset):

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
        num_rows_per_batch=8,
        split_frame_length=2000,
        is_randomize_tokens=True,
        is_deterministic=False,
        use_tf_spectral_ops=True
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
            num_rows_per_batch=num_rows_per_batch,
            split_frame_length=split_frame_length,
            is_randomize_tokens=is_randomize_tokens,
            is_deterministic=is_deterministic,
            use_tf_spectral_ops=use_tf_spectral_ops
        )

    def _extract_target_sequence_with_indices(self, features, state_events_end_token=None):
        """Extract target sequence corresponding to audio token segment."""
        # ===== current segment tokens ===== #
        target_start_idx = features['input_event_start_indices'][0]
        target_end_idx = features['input_event_end_indices'][-1]

        targets_orig = features['targets'][:]
        features['targets'] = targets_orig[target_start_idx:target_end_idx]

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

        # ===== prev segment tokens ===== #
        if 'input_event_start_indices_prev' in features:
            target_start_idx_prev = features['input_event_start_indices_prev'][0]
            target_end_idx_prev = features['input_event_end_indices_prev'][-1]
            features['targets_prev'] = targets_orig[target_start_idx_prev:target_end_idx_prev]

            if state_events_end_token is not None:
                # Extract the state events corresponding to the audio start token, and
                # prepend them to the targets array.
                state_event_start_idx_prev = features['input_state_event_indices_prev'][0]
                state_event_end_idx_prev = state_event_start_idx_prev + 1

                while features['state_events'][
                        state_event_end_idx_prev - 1] != state_events_end_token:
                    state_event_end_idx_prev += 1
                            
                features['targets_prev'] = np.concatenate([
                    features['state_events'][state_event_start_idx_prev:state_event_end_idx_prev],
                    features['targets_prev']
                ], axis=0)
        else:
            features['targets_prev'] = np.array([1131, 1])

        return features

    def _pad_length(self, row):
        inputs = row['inputs'][:self.mel_length].to(torch.float32)

        # current segment targets
        targets = torch.from_numpy(row['targets'][:self.event_length]).to(torch.long)
        targets = targets + self.vocab.num_special_tokens()

        # prev segment targets
        targets_prev = torch.from_numpy(row['targets_prev'][:self.event_length]).to(torch.long)
        targets_prev = targets_prev + self.vocab.num_special_tokens()

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
        
        if targets_prev.shape[0] < self.event_length:
            eos = torch.ones(1, dtype=targets_prev.dtype, device=targets_prev.device)
            if self.event_length - targets_prev.shape[0] - 1 > 0:
                pad = torch.ones(self.event_length - targets_prev.shape[0] - 1, dtype=targets_prev.dtype, device=targets_prev.device) * -100
                targets_prev = torch.cat([targets_prev, eos, pad], dim=0)
            else:
                targets_prev = torch.cat([targets_prev, eos], dim=0)
        
        return {
            'inputs': inputs, 
            'targets': targets, 
            'targets_prev': targets_prev,
            "input_times": row["input_times"]
        }
    
    def _random_chunk(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        if self.is_deterministic:
            start_length = 16
        else:
            # NOTE: this is temporary, start_length must be at least 1 mel_length later
            start_length = random.randint(0, random_length)
            start_length_prev = start_length - self.mel_length
        for k in row.keys():
            if k in ['inputs', 'input_times', 'input_event_start_indices', 'input_event_end_indices', 'input_state_event_indices']:
                new_row[k] = row[k][start_length:start_length+self.mel_length]

                if start_length_prev > 0:
                    new_row[k + '_prev'] = row[k][start_length_prev:start_length_prev+self.mel_length]
            else:
                new_row[k] = row[k]

        return new_row
    
    def __getitem__(self, idx):
        ns, audio, inst_names = self._preprocess_inputs(self.df[idx])
        
        row = self._tokenize(ns, audio, inst_names)

        # NOTE: if we need to ensure that the chunks in `rows` to be contiguous, 
        # use `length = self.mel_length` in `_split_frame`:
        rows = self._split_frame(row, length=self.split_frame_length)
        
        inputs, targets, targets_prev, frame_times, num_insts = [], [], [], [], []
        if len(rows) > self.num_rows_per_batch:
            if self.is_deterministic:
                start_idx = 2
            else:
                start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx : start_idx + self.num_rows_per_batch]
        
        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            row = self._run_length_encode_shifts(row, feature_key="targets")
            row = self._run_length_encode_shifts(row, feature_key="targets_prev")
            
            row = self._compute_spectrogram(row)

            # -- random order augmentation --
            if self.is_randomize_tokens:
                t = self.randomize_tokens([self.get_token_name(t) for t in row["targets"]])
                t = np.array([self.token_to_idx(k) for k in t])
                t = self._remove_redundant_tokens(t)
                row["targets"] = t

                t = self.randomize_tokens([self.get_token_name(t) for t in row["targets_prev"]])
                t = np.array([self.token_to_idx(k) for k in t])
                t = self._remove_redundant_tokens(t)
                row["targets_prev"] = t
                        
            row = self._pad_length(row)
            inputs.append(row["inputs"])
            targets.append(row["targets"])
            targets_prev.append(row["targets_prev"]) 

        return torch.stack(inputs), torch.stack(targets), torch.stack(targets_prev)


def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    targets_prev = torch.cat([k[2] for k in lst])

    return inputs, targets, targets_prev

if __name__ == '__main__':
    dataset = SlakhDatasetWithPrevSegmem(
        root_dir='/data/slakh2100_flac_redux/test/',
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256,
        split_frame_length=2000,
        is_deterministic=False,
        is_randomize_tokens=False
    )
    print("pitch", dataset.codec.event_type_range("pitch"))
    print("velocity", dataset.codec.event_type_range("velocity"))
    print("tie", dataset.codec.event_type_range("tie"))
    print("program", dataset.codec.event_type_range("program"))
    print("drum", dataset.codec.event_type_range("drum"))
    dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
    for idx, item in enumerate(dl):
        inputs, targets, targets_prev = item
        for i in range(len(targets)):
            print(i, targets[i][:100], targets_prev[i][:100])
            print("===")
        break
    