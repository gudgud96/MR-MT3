from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from immutabledict import immutabledict
from torch.utils.data import Dataset, DataLoader
from contrib import event_codec,spectrograms, vocabularies, note_sequences, metrics_utils, run_length_encoding
import numpy as np
import librosa
import note_seq


_SLAKH_CLASS_PROGRAMS = immutabledict({
    'Acoustic Piano': 0,
    'Electric Piano': 4,
    'Chromatic Percussion': 8,
    'Organ': 16,
    'Acoustic Guitar': 24,
    'Clean Electric Guitar': 26,
    'Distorted Electric Guitar': 29,
    'Acoustic Bass': 32,
    'Electric Bass': 33,
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'Orchestral Harp': 46,
    'Timpani': 47,
    'String Ensemble': 48,
    'Synth Strings': 50,
    'Choir and Voice': 52,
    'Orchestral Hit': 55,
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'French Horn': 60,
    'Brass Section': 61,
    'Soprano/Alto Sax': 64,
    'Tenor Sax': 66,
    'Baritone Sax': 67,
    'Oboe': 68,
    'English Horn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    'Pipe': 73,
    'Synth Lead': 80,
    'Synth Pad': 88
})


def slakh_class_to_program_and_is_drum(slakh_class: str) -> Tuple[int, bool]:
  """Map Slakh class string to program number and boolean indicating drums."""
  if slakh_class == 'Drums':
    return 0, True
  elif slakh_class not in _SLAKH_CLASS_PROGRAMS:
    raise ValueError('unknown Slakh class: %s' % slakh_class)
  else:
    return _SLAKH_CLASS_PROGRAMS[slakh_class], False


class PitchBendError(Exception):
  pass


def audio_to_frames(audio, spectrogram_config):
    """Compute spectrogram frames from audio."""
    frame_size = spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio = np.pad(audio, padding, mode='constant')
    frames = spectrograms.split_audio(audio, spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / \
        spectrogram_config.frames_per_second
    return frames, times


def add_track_to_notesequence(ns: note_seq.NoteSequence,
                              track: note_seq.NoteSequence,
                              program: int, is_drum: bool,
                              ignore_pitch_bends: bool):
    """Add a track to a NoteSequence."""
    if track.pitch_bends and not ignore_pitch_bends:
        raise PitchBendError
    track_sus = note_seq.apply_sustain_control_changes(track)
    for note in track_sus.notes:
        note.program = program
        note.is_drum = is_drum
        ns.notes.extend([note])
        ns.total_time = max(ns.total_time, note.end_time)
    

def tokenize_slakh_example(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]] = None,
    ignore_pitch_bends: bool = True
) -> tf.data.Dataset:
  """Tokenize a Slakh multitrack note transcription example."""
  



class SlakhMT3Dataset(Dataset):
    def __init__(self, is_training_data, onsets_only, include_ties, track_specs, ignore_pitch_bends):
        self.spectrogram_config = None
        self.codec = None
        self.is_training_data = is_training_data
        self.onsets_only = onsets_only
        self.include_ties = include_ties
        self.track_specs = track_specs
        self.ignore_pitch_bends = ignore_pitch_bends
    
    def __getitem__
    
    def tokenize(
        self,
        sequences, 
        samples, 
        sample_rate, 
        inst_names, 
        example_id
    ):
        if sample_rate != self.spectrogram_config.sample_rate:
            samples = librosa.resample(
                samples, sample_rate, self.spectrogram_config.sample_rate)

        frames, frame_times = audio_to_frames(samples, self.spectrogram_config)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        assert len(tracks) == len(inst_names)
        if self.track_specs:
            # Specific tracks expected.
            assert len(tracks) == len(self.track_specs)
            for track, spec, inst_name in zip(tracks, self.track_specs, inst_names):
                # Make sure the instrument name matches what we expect.
                assert inst_name.decode() == spec.name
                try:
                    add_track_to_notesequence(ns, track,
                                            program=spec.program, is_drum=spec.is_drum,
                                            ignore_pitch_bends=self.ignore_pitch_bends)
                except PitchBendError:
                    # TODO(iansimon): is there a way to count these?
                    return
        else:
            for track, inst_name in zip(tracks, inst_names):
                # Instrument name should be Slakh class.
                program, is_drum = slakh_class_to_program_and_is_drum(
                    inst_name.decode())
                try:
                    add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                            ignore_pitch_bends=self.ignore_pitch_bends)
                except PitchBendError:
                    # TODO(iansimon): is there a way to count these?
                    return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        if self.is_training_data:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

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
            'inputs': frames,
            'input_times': frame_times,
            'targets': events,
            'input_event_start_indices': event_start_indices,
            'input_event_end_indices': event_end_indices,
            'state_events': state_events,
            'input_state_event_indices': state_event_indices,
            'sequence': ns.SerializeToString()
        }