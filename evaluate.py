import mir_eval
import glob
import pretty_midi
import os
import numpy as np
import librosa
import copy
import note_seq
from contrib import note_sequences, event_codec, vocabularies
import collections


def get_granular_program(program_number, is_drum, granularity_type):
    if granularity_type == "full":
        return program_number
    elif granularity_type == "midi_class":
        return (program_number // 8) * 8
    elif granularity_type == "flat":
        return 0 if not is_drum else 1

def mt3_program_aware_note_scores_v2(
    ref_mid, est_mid, granularity_type
):

    ref_ns = note_seq.midi_file_to_note_sequence(ref_mid)
    est_ns = note_seq.midi_file_to_note_sequence(est_mid)
    program_map_fn = vocabularies.PROGRAM_GRANULARITIES[
      granularity_type].program_map_fn

    ref_ns = copy.deepcopy(ref_ns)
    for note in ref_ns.notes:
        if not note.is_drum:
            note.program = program_map_fn(note.program)

    est_ns = copy.deepcopy(est_ns)
    for note in est_ns.notes:
        if not note.is_drum:
            note.program = program_map_fn(note.program)

    program_and_is_drum_tuples = (
        set((note.program, note.is_drum) for note in ref_ns.notes) |
        set((note.program, note.is_drum) for note in est_ns.notes)
    )

    drum_precision_sum = 0.0
    drum_precision_count = 0
    drum_recall_sum = 0.0
    drum_recall_count = 0

    nondrum_precision_sum = 0.0
    nondrum_precision_count = 0
    nondrum_recall_sum = 0.0
    nondrum_recall_count = 0

    for program, is_drum in program_and_is_drum_tuples:
        est_track = note_sequences.extract_track(est_ns, program, is_drum)
        ref_track = note_sequences.extract_track(ref_ns, program, is_drum)

        est_intervals, est_pitches, unused_est_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_track))
        ref_intervals, ref_pitches, unused_ref_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_track))

        args = {
            'ref_intervals': ref_intervals, 'ref_pitches': ref_pitches,
            'est_intervals': est_intervals, 'est_pitches': est_pitches
        }
        if is_drum:
            args['offset_ratio'] = None

        precision, recall, unused_f_measure, unused_avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(**args))

        if is_drum:
            drum_precision_sum += precision * len(est_intervals)
            drum_precision_count += len(est_intervals)
            drum_recall_sum += recall * len(ref_intervals)
            drum_recall_count += len(ref_intervals)
        else:
            nondrum_precision_sum += precision * len(est_intervals)
            nondrum_precision_count += len(est_intervals)
            nondrum_recall_sum += recall * len(ref_intervals)
            nondrum_recall_count += len(ref_intervals)

    precision_sum = drum_precision_sum + nondrum_precision_sum
    precision_count = drum_precision_count + nondrum_precision_count
    recall_sum = drum_recall_sum + nondrum_recall_sum
    recall_count = drum_recall_count + nondrum_recall_count

    precision = (precision_sum / precision_count) if precision_count else 0
    recall = (recall_sum / recall_count) if recall_count else 0
    f_measure = mir_eval.util.f_measure(precision, recall)

    drum_precision = ((drum_precision_sum / drum_precision_count)
                        if drum_precision_count else 0)
    drum_recall = ((drum_recall_sum / drum_recall_count)
                    if drum_recall_count else 0)
    drum_f_measure = mir_eval.util.f_measure(drum_precision, drum_recall)

    nondrum_precision = ((nondrum_precision_sum / nondrum_precision_count)
                        if nondrum_precision_count else 0)
    nondrum_recall = ((nondrum_recall_sum / nondrum_recall_count)
                        if nondrum_recall_count else 0)
    nondrum_f_measure = mir_eval.util.f_measure(nondrum_precision, nondrum_recall)

    return {
        f'Onset + offset + program precision ({granularity_type})': precision,
        f'Onset + offset + program recall ({granularity_type})': recall,
        f'Onset + offset + program F1 ({granularity_type})': f_measure,
        f'Drum onset precision ({granularity_type})': drum_precision,
        f'Drum onset recall ({granularity_type})': drum_recall,
        f'Drum onset F1 ({granularity_type})': drum_f_measure,
        f'Nondrum onset + offset + program precision ({granularity_type})':
            nondrum_precision,
        f'Nondrum onset + offset + program recall ({granularity_type})':
            nondrum_recall,
        f'Nondrum onset + offset + program F1 ({granularity_type})':
            nondrum_f_measure
    }


def mt3_program_aware_note_scores(
    ref_mid, est_mid, granularity_type
):
    # group notes by program number
    ref_inst_to_notes_mapping = {}
    est_inst_to_notes_mapping = {}

    # following MT3, this will group notes under the same instrument program, determined by the granularity
    for inst in ref_mid.instruments:
        cur_ref_program = get_granular_program(inst.program, inst.is_drum, granularity_type)
        if (cur_ref_program, inst.is_drum) in ref_inst_to_notes_mapping:
            ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] += [note for note in inst.notes]
        else:
            ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] = [note for note in inst.notes]

    for inst in est_mid.instruments:
        cur_est_program = get_granular_program(inst.program, inst.is_drum, granularity_type)
        if (cur_est_program, inst.is_drum) in est_inst_to_notes_mapping:
            est_inst_to_notes_mapping[(cur_est_program, inst.is_drum)] += [note for note in inst.notes]
        else:
            est_inst_to_notes_mapping[(cur_est_program, inst.is_drum)] = [note for note in inst.notes]
    
    program_and_is_drum_tuples = set(ref_inst_to_notes_mapping.keys()) | set(est_inst_to_notes_mapping.keys())
    drum_precision_sum = 0.0
    drum_precision_count = 0
    drum_recall_sum = 0.0
    drum_recall_count = 0

    nondrum_precision_sum = 0.0
    nondrum_precision_count = 0
    nondrum_recall_sum = 0.0
    nondrum_recall_count = 0

    for program, is_drum in program_and_is_drum_tuples:
        if (program, is_drum) in ref_inst_to_notes_mapping:
            ref_notes = ref_inst_to_notes_mapping[(program, is_drum)]
            ref_intervals = np.array([[note.start, note.end] for note in ref_notes])
            ref_pitches = np.array([librosa.midi_to_hz(note.pitch) for note in ref_notes])
        else:
            # ref does not have this instrument
            ref_intervals = np.zeros((0, 2))
            ref_pitches = np.zeros(0)
        
        if (program, is_drum) in est_inst_to_notes_mapping:
            est_notes = est_inst_to_notes_mapping[(program, is_drum)]
            est_intervals = np.array([[note.start, note.end] for note in est_notes])
            est_pitches = np.array([librosa.midi_to_hz(note.pitch) for note in est_notes])
        else:
            # est does not have this instrument
            est_intervals = np.zeros((0, 2))
            est_pitches = np.zeros(0)
    
        args = {
            'ref_intervals': ref_intervals, 'ref_pitches': ref_pitches,
            'est_intervals': est_intervals, 'est_pitches': est_pitches
        }
        if is_drum:
            args['offset_ratio'] = None
        precision, recall, unused_f_measure, unused_avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(**args))

        if is_drum:
            drum_precision_sum += precision * len(est_intervals)
            drum_precision_count += len(est_intervals)
            drum_recall_sum += recall * len(ref_intervals)
            drum_recall_count += len(ref_intervals)
        else:
            nondrum_precision_sum += precision * len(est_intervals)
            nondrum_precision_count += len(est_intervals)
            nondrum_recall_sum += recall * len(ref_intervals)
            nondrum_recall_count += len(ref_intervals)
    
    precision_sum = drum_precision_sum + nondrum_precision_sum
    precision_count = drum_precision_count + nondrum_precision_count
    recall_sum = drum_recall_sum + nondrum_recall_sum
    recall_count = drum_recall_count + nondrum_recall_count

    precision = (precision_sum / precision_count) if precision_count else 0
    recall = (recall_sum / recall_count) if recall_count else 0
    f_measure = mir_eval.util.f_measure(precision, recall)

    drum_precision = ((drum_precision_sum / drum_precision_count)
                        if drum_precision_count else 0)
    drum_recall = ((drum_recall_sum / drum_recall_count)
                    if drum_recall_count else 0)
    drum_f_measure = mir_eval.util.f_measure(drum_precision, drum_recall)

    nondrum_precision = ((nondrum_precision_sum / nondrum_precision_count)
                        if nondrum_precision_count else 0)
    nondrum_recall = ((nondrum_recall_sum / nondrum_recall_count)
                        if nondrum_recall_count else 0)
    nondrum_f_measure = mir_eval.util.f_measure(nondrum_precision, nondrum_recall)

    return {
        f'Onset + offset + program precision ({granularity_type})': precision,
        f'Onset + offset + program recall ({granularity_type})': recall,
        f'Onset + offset + program F1 ({granularity_type})': f_measure,
        f'Drum onset precision ({granularity_type})': drum_precision,
        f'Drum onset recall ({granularity_type})': drum_recall,
        f'Drum onset F1 ({granularity_type})': drum_f_measure,
        f'Nondrum onset + offset + program precision ({granularity_type})':
            nondrum_precision,
        f'Nondrum onset + offset + program recall ({granularity_type})':
            nondrum_recall,
        f'Nondrum onset + offset + program F1 ({granularity_type})':
            nondrum_f_measure
    }
    # })
    # print("{:.4} {:.4} {:.4}".format(f_measure, drum_f_measure, nondrum_f_measure))
    # return f_measure


def loop_transcription_eval(ref_mid, est_mid):
    """
    This evaluation takes in account the separability of the model. Goes by "track" instead of tight
    coupling to "program number". This is because of a few reasons:
    - for loops, the program number in ref can be arbitrary
        - e.g. how do you assign program number to Vox?
        - no one use program number for synth / sampler etc.
        - string contrabass VS bass midi class are different, but can be acceptable
        - leads and key / synth pads and electric piano
    - the "track splitting" aspect is more important than the accuracy of the midi program number
        - we can have wrong program number, but as long as they are grouped in the correct track
    - hence we propose 2 more evaluation metrics:
        - f1_score_matrix for each ref_track VS est_track, take the mean of the maximum f1 score for each ref_track
        - number of tracks
    """
    score_matrix = np.zeros((len(ref_mid.instruments), len(est_mid.instruments)))

    for i, ref_inst in enumerate(ref_mid.instruments):
        for j, est_inst in enumerate(est_mid.instruments):
            if ref_inst.is_drum == est_inst.is_drum:
                ref_intervals = np.array([[note.start, note.end] for note in ref_inst.notes])
                ref_pitches = np.array([librosa.midi_to_hz(note.pitch) for note in ref_inst.notes])
                est_intervals = np.array([[note.start, note.end] for note in est_inst.notes])
                est_pitches = np.array([librosa.midi_to_hz(note.pitch) for note in est_inst.notes])

                _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches)
                score_matrix[i][j] = f1
    
    inst_idx = np.argmax(score_matrix, axis=-1)
    ref_progs = [inst.program for inst in ref_mid.instruments]
    est_progs = [est_mid.instruments[idx].program for idx in inst_idx]
    return np.mean(np.max(score_matrix, axis=-1)), len(ref_mid.instruments), len(est_mid.instruments)
    

dir = sorted(glob.glob("loops_data/*/mix.mid"))
from tqdm import tqdm
# dir = sorted(glob.glob("babyslakh_16k/*/all_src.mid"))
# dir = sorted(glob.glob("/data/slakh2100_flac_redux/test/*/all_src.mid"))
ref_sum, est_sum = [], []

flat, midi_class, full, new, inst_num_ref, inst_num_est = 0, 0, 0, 0, 0, 0
scores = collections.defaultdict(list)
for elem in tqdm(dir):
    name = elem.split("/")[-2]
    # print(name)
    # ref_mid = pretty_midi.PrettyMIDI(elem)
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("loops_out", name, "mix.mid"))
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("babyslakh_16k_out", name, "mix.mid"))
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("/data/mt3_recon/slakh2100_flac_redux_official", name, "mix.mid"))

    for granularity in ["flat", "midi_class", "full"]:
        for score_name, score in mt3_program_aware_note_scores_v2(
            elem, 
            # os.path.join("/data/mt3_recon/slakh2100_flac_redux_official", name, "mix.mid"), 
            os.path.join("loops_out", name, "mix.mid"),
            granularity_type=granularity
        ).items():
            scores[score_name].append(score)

mean_scores = {k: np.mean(v) for k, v in scores.items()}
for key in sorted(list(mean_scores)):
    print("{}: {:.4}".format(key, mean_scores[key]))
#     # mean_f1, num_ref, num_est = loop_transcription_eval(ref_mid, est_mid)
#     # new += mean_f1
#     # inst_num_ref += num_ref
#     # inst_num_est += num_est


# print("flat={:.4} midi_class={:.4} full={:.4} new={:.4} inst_ref={:.4} inst_est={:.4}".format(
#     flat / len(dir),
#     midi_class / len(dir),
#     full / len(dir),
#     # new / len(dir),
#     # inst_num_ref / len(dir),
#     # inst_num_est / len(dir)
# ))

# # print("ref inst num: ", np.mean(ref_sum), np.std(ref_sum))
# # print("est inst num: ", np.mean(est_sum), np.std(est_sum))

