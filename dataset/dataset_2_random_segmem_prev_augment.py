import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from dataset.dataset_2_random_segmem_prev import SlakhDatasetWithPrevSegmem

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

class SlakhDatasetWithPrevSegmemAugment(SlakhDatasetWithPrevSegmem):

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
        use_tf_spectral_ops=True,
        prev_augment_frames=1,
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
        self.prev_augment_frames = prev_augment_frames
   
    def _random_chunk(self, row):
        new_row = {}
        input_length = row['inputs'].shape[0]
        random_length = input_length - self.mel_length
        if random_length < 1:
            return row
        if self.is_deterministic:
            start_length = 16
        else:
            start_length = random.randint(0, random_length)

            # NOTE: augmentation 1: we don't just get context only from the previous segment (`mel_length` frames before),
            # but we can get it up to N segments before. The hypothesis is that close enough segments can alsp give context.
            prev_segment_index = random.randint(1, self.prev_augment_frames)
            start_length_prev = start_length - prev_segment_index * self.mel_length
        
        for k in row.keys():
            if k in ['inputs', 'input_times', 'input_event_start_indices', 'input_event_end_indices', 'input_state_event_indices']:
                new_row[k] = row[k][start_length:start_length+self.mel_length]

                # NOTE: with augmentation 1, we might have more cases that result in empty context
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
    dataset = SlakhDatasetWithPrevSegmemAugment(
        root_dir='/data/slakh2100_flac_redux/test/',
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256,
        split_frame_length=2000,
        is_deterministic=False,
        is_randomize_tokens=False,
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
    