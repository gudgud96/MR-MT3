import torch
import json
import random
import librosa
import note_seq
from glob import glob
import numpy as np

from dataset.dataset_2_random_segmem_prev import SlakhDatasetWithPrevSegmem
from dataset.dataset_2_random import SlakhDataset

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

class SlakhStemMixDataset(SlakhDatasetWithPrevSegmem):

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
        num_rows_per_batch=8,
        split_frame_length=2000,
        is_randomize_tokens=True,
        is_deterministic=False,
        use_tf_spectral_ops=True,
        prev_augment_frames=1,        
    ) -> None:
        self.stems_folder = stems_folder
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

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/**/{self.audio_filename}'))
        for a_f in audio_files:
            # get path for stems
            stems_path = a_f.replace(self.audio_filename, self.stems_folder)
            stem_audio_files = sorted(glob(f'{stems_path}/*_16k.wav'))
            stem_midi_files = [k.replace(self.stems_folder, self.midi_folder).replace("_16k.wav", ".mid") for k in stem_audio_files]

            inst_path = a_f.replace(self.audio_filename, self.inst_filename)
            with open(inst_path) as f:
                inst_names = json.load(f)
            df.append({'inst_names': inst_names, 'audio_path': stem_audio_files, 'midi_path': stem_midi_files})

        assert len(df) > 0
        print('total file:', len(df))
        if shuffle:
            random.shuffle(df)
        return df

    def _parse_midi(self, midi_paths):
        note_seqs = []
        for midi_path in midi_paths:
            # this can be pretty_midi.PrettyMIDI() obj / string path to midi
            note_seqs.append(note_seq.midi_file_to_note_sequence(midi_path))
        return note_seqs

    def _parse_audio(self, audio_paths):
        mix = None
        for path in audio_paths:
            audio, sr = librosa.load(path, sr=None)
            if sr != self.spectrogram_config.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.spectrogram_config.sample_rate)
            if mix is None:
                mix = audio
            else:
                mix += audio
        return mix

    def _random_stem_mix(self, row, num_stems_to_mix=None):
        if num_stems_to_mix is None:
            num_stems_to_mix = random.randint(1, len(row['audio_path']))
        else:
            assert num_stems_to_mix > 0
            num_stems_to_mix = min(num_stems_to_mix, len(row['audio_path']))

        chosen_audio_stems = sorted(random.sample(row['audio_path'], k=num_stems_to_mix))
        chosen_midi_stems = [k.replace(self.stems_folder, self.midi_folder).replace("_16k.wav", ".mid") for k in chosen_audio_stems]
        instrument_keys = [k.split('/')[-1].replace('_16k.wav', '') for k in chosen_audio_stems]
        instrument_values = [row['inst_names'][k] for k in instrument_keys]

        return chosen_audio_stems, chosen_midi_stems, instrument_values

    def _preprocess_inputs(self, row):
        chosen_audio_stems, chosen_midi_stems, inst_names = self._random_stem_mix(
            row,
            num_stems_to_mix=None
        )
        ns = self._parse_midi(chosen_midi_stems)
        audio = self._parse_audio(chosen_audio_stems)

        return ns, audio, inst_names
    
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
            # print('prev_segment_index', prev_segment_index, self.prev_augment_frames)
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

        # NOTE: by default, this is self._split_frame(row, length=2000)
        # this does not guarantee the chunks in `rows` to be contiguous.
        # if we need to ensure that the chunks in `rows` to be contiguous, use:
        rows = self._split_frame(row, length=self.split_frame_length)
        # rows = self._split_frame(row)
        
        inputs, targets, targets_prev, frame_times, num_insts = [], [], [], [], []
        if len(rows) > self.num_rows_per_batch:
            if self.is_deterministic:
                start_idx = 2
            else:
                start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx : start_idx + self.num_rows_per_batch]
        
        predictions = []
        predictions_prev = []
        wavs = []
        
        fake_start = None
        for j, row in enumerate(rows):
            row = self._random_chunk(row)
            row = self._extract_target_sequence_with_indices(row, self.tie_token)
            row = self._run_length_encode_shifts(row, feature_key="targets")
            row = self._run_length_encode_shifts(row, feature_key="targets_prev")
            
            # wavs.append(row["inputs"].reshape(-1,))
            # sf.write(f"test_{j}.wav", row["inputs"].reshape(-1,), 16000, "PCM_24")

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
            
            # print(j, [self.get_token_name(t) for t in row["targets"]])
            
            row = self._pad_length(row)
            inputs.append(row["inputs"])
            targets.append(row["targets"])
            targets_prev.append(row["targets_prev"]) 

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

            # result_prev = row["targets_prev"]
            # EOS_TOKEN_ID = 1    # TODO: this is a hack!
            # after_eos = torch.cumsum(
            #     (result_prev == EOS_TOKEN_ID).float(), dim=-1
            # )
            # result_prev -= self.vocab.num_special_tokens()
            # result_prev = torch.where(after_eos.bool(), -1, result_prev)

            # print("start_times", row["input_times"][0])
            # if fake_start is None:
            #     fake_start = row["input_times"][0]
            # # predictions = []
            # predictions_prev.append({
            #     'est_tokens': result_prev.cpu().detach().numpy(),    # has to be numpy here, or else problematic
            #     # 'start_time': row["input_times"][0] - fake_start,
            #     'start_time': j * 2.048,
            #     # 'start_time': 0,
            #     'raw_inputs': []
            # })
            # ========== for reconstructing the MIDI from events =========== #
        
        # ========== for reconstructing the MIDI from events =========== #
        # encoding_spec = note_sequences.NoteEncodingWithTiesSpec
        # result = metrics_utils.event_predictions_to_ns(
        #     predictions, codec=self.codec, encoding_spec=encoding_spec)
        # note_seq.sequence_proto_to_midi_file(result['est_ns'], "test_out.mid")   
        # sf.write(f"test_out.wav", np.concatenate(wavs), 16000, "PCM_24")

        # result_prev = metrics_utils.event_predictions_to_ns(
        #     predictions_prev, codec=self.codec, encoding_spec=encoding_spec)
        # note_seq.sequence_proto_to_midi_file(result_prev['est_ns'], "test_out_prev.mid")   
        # ========== for reconstructing the MIDI from events =========== #  

        return torch.stack(inputs), torch.stack(targets), torch.stack(targets_prev)    


def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    targets_prev = torch.cat([k[2] for k in lst])

    return inputs, targets, targets_prev


if __name__ == '__main__':
    dataset = SlakhStemMixDataset(
        root_dir='/data/slakh2100_flac_redux/test/',
        shuffle=False,
        is_train=False,
        include_ties=True,
        mel_length=256
    )
    for item in dataset:
        inputs, targets = item
        print(inputs.shape, targets)
        break

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
