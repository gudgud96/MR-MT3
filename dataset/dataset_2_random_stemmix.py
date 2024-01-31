import torch
import json
import random
import librosa
import note_seq
from glob import glob

from dataset.dataset_2_random import SlakhDataset

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
        num_rows_per_batch=8,
        split_frame_length=2000,
        is_randomize_tokens=True,
        is_deterministic=False,
        use_tf_spectral_ops=True
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

    def _build_dataset(self, root_dir, shuffle=True):
        df = []
        audio_files = sorted(glob(f'{root_dir}/**/{self.audio_filename}'))
        for a_f in audio_files:
            # get path for stems
            stems_path = a_f.replace(self.audio_filename, self.stems_folder)
            stem_audio_files = sorted(glob(f'{stems_path}/*.flac'))
            stem_midi_files = [k.replace(self.stems_folder, self.midi_folder).replace(".flac", ".mid") for k in stem_audio_files]

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
        chosen_midi_stems = [k.replace(self.stems_folder, self.midi_folder).replace(".flac", ".mid") for k in chosen_audio_stems]
        instrument_keys = [k.split('/')[-1].replace('.flac', '') for k in chosen_audio_stems]
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


def collate_fn(lst):
    inputs = torch.cat([k[0] for k in lst])
    targets = torch.cat([k[1] for k in lst])
    return inputs, targets


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
