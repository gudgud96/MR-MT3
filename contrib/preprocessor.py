# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transcription preprocessors."""

from typing import Tuple

from immutabledict import immutabledict
import note_seq


def guitarset_instrument_to_program(instrument: str) -> int:
    """GuitarSet is all guitar, return the first MIDI guitar program."""
    if instrument == 'Clean Guitar':
        return 24
    else:
        raise ValueError('Unknown GuitarSet instrument: %s' % instrument)


_URMP_INSTRUMENT_PROGRAMS = immutabledict({
    'vn': 40,   # violin
    'va': 41,   # viola
    'vc': 42,   # cello
    'db': 43,   # double bass
    'tpt': 56,  # trumpet
    'tbn': 57,  # trombone
    'tba': 58,  # tuba
    'hn': 60,   # French horn
    'sax': 64,  # saxophone
    'ob': 68,   # oboe
    'bn': 70,   # bassoon
    'cl': 71,   # clarinet
    'fl': 73    # flute
})

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
