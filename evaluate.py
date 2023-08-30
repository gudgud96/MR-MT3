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
import concurrent.futures
import traceback


def get_granular_program(program_number, is_drum, granularity_type):
    if granularity_type == "full":
        return program_number
    elif granularity_type == "midi_class":
        return (program_number // 8) * 8
    elif granularity_type == "flat":
        return 0 if not is_drum else 1


def compute_transcription_metrics(ref_mid, est_mid):
  """Helper function to compute onset/offset, onset only, and frame metrics."""
  ns_ref = note_seq.midi_file_to_note_sequence(ref_mid)
  ns_est = note_seq.midi_file_to_note_sequence(est_mid)
  intervals_ref, pitches_ref, _ = note_seq.sequences_lib.sequence_to_valued_intervals(ns_ref)
  intervals_est, pitches_est, _ = note_seq.sequences_lib.sequence_to_valued_intervals(ns_est)
  len_est_intervals = len(intervals_est)
  len_ref_intervals = len(intervals_ref)

  # onset-offset
  onoff_precision, onoff_recall, onoff_f1, onoff_overlap = mir_eval.transcription.precision_recall_f1_overlap(
    intervals_ref, pitches_ref, intervals_est, pitches_est)

  # onset-only
  on_precision, on_recall, on_f1, on_overlap = mir_eval.transcription.precision_recall_f1_overlap(
    intervals_ref, pitches_ref, intervals_est, pitches_est, offset_ratio=None)

  return {
      'len_ref_intervals': len_ref_intervals, 
      'len_est_intervals': len_est_intervals,
      'onoff_precision': onoff_precision, 
      'onoff_recall': onoff_recall, 
      'onoff_f1': onoff_f1, 
      'onoff_overlap': onoff_overlap, 
      'on_precision': on_precision, 
      'on_recall': on_recall, 
      'on_f1': on_f1, 
      'on_overlap': on_overlap,
  }


def mt3_program_aware_note_scores_v2(
    ref_mid, est_mid, granularity_type, offset=False
):
    res = dict()
    ref_ns = note_seq.midi_file_to_note_sequence(ref_mid)
    est_ns = note_seq.midi_file_to_note_sequence(est_mid)
    
    def remove_drums(ns):
      ns_drumless = note_seq.NoteSequence()
      ns_drumless.CopyFrom(ns)
    #   del ns_drumless.notes[:]
    #   ns_drumless.notes.extend([note for note in ns.notes if not note.is_drum])
      return ns_drumless

    est_ns_drumless = remove_drums(est_ns)
    ref_ns_drumless = remove_drums(ref_ns)

    # Whether or not there are separate tracks, compute metrics for the full
    # NoteSequence minus drums.
    est_tracks = [est_ns_drumless]
    ref_tracks = [ref_ns_drumless]
    use_track_offsets = [False]
    use_track_velocities = [False]
    track_instrument_names = ['']

    for est_ns, ref_ns, use_offsets, use_velocities, instrument_name in zip(
        est_tracks, ref_tracks, use_track_offsets, use_track_velocities,
        track_instrument_names):

      est_intervals, est_pitches, est_velocities = (
          note_seq.sequences_lib.sequence_to_valued_intervals(est_ns))

      ref_intervals, ref_pitches, ref_velocities = (
          note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns))

      # Precision / recall / F1 using onsets (and pitches) only.
      precision, recall, f_measure, avg_overlap_ratio = (
          mir_eval.transcription.precision_recall_f1_overlap(
              ref_intervals=ref_intervals,
              ref_pitches=ref_pitches,
              est_intervals=est_intervals,
              est_pitches=est_pitches,
              offset_ratio=None))
      res['Onset precision'] = precision
      res['Onset recall'] = recall
      res['Onset F1'] = f_measure

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

    # print(granularity_type, 'ref', set((note.program, note.is_drum) for note in ref_ns.notes))
    # print(granularity_type, 'est', set((note.program, note.is_drum) for note in est_ns.notes))
    program_and_is_drum_tuples = (
        set((note.program, note.is_drum) for note in ref_ns.notes) |
        set((note.program, note.is_drum) for note in est_ns.notes)
    )
    # print(granularity_type, 'program', program_and_is_drum_tuples)

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
        
        # print(program, is_drum)
        # print(est_intervals[:10])
        # print(ref_intervals[:10])
        # print("----")
        # print(est_pitches[:10])
        # print(ref_pitches[:10])
        # print("....")

        args = {
            'ref_intervals': ref_intervals, 'ref_pitches': ref_pitches,
            'est_intervals': est_intervals, 'est_pitches': est_pitches
        }
        if offset:
            if is_drum:
                args['offset_ratio'] = None
        else:
            args['offset_ratio'] = None
            

        precision, recall, unused_f_measure, unused_avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(**args))
        
        # print('precision', precision)
        # print('recall', recall)
        # print("=====")

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

    res.update({
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
    })

    return res


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
    
        # NOTE: like Perceiver, disable offset calculation
        args = {
            'ref_intervals': ref_intervals, 'ref_pitches': ref_pitches,
            'est_intervals': est_intervals, 'est_pitches': est_pitches,
            'offset_ratio': None
        }
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
        f'Onset + program precision ({granularity_type})': precision,
        f'Onset + program recall ({granularity_type})': recall,
        f'Onset + program F1 ({granularity_type})': f_measure,
        f'Drum onset precision ({granularity_type})': drum_precision,
        f'Drum onset recall ({granularity_type})': drum_recall,
        f'Drum onset F1 ({granularity_type})': drum_f_measure,
        f'Nondrum onset + program precision ({granularity_type})':
            nondrum_precision,
        f'Nondrum onset + program recall ({granularity_type})':
            nondrum_recall,
        f'Nondrum onset + program F1 ({granularity_type})':
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
    

from tqdm import tqdm
# Loop data
# dir = sorted(glob.glob("loops_data/*/mix.mid"))
# dir2 = [k.replace("loops_data", "loops_out") for k in dir]

# Slakh
fname = "slakh_recon_randomorder"
# fname = "slakh_recon_norm_v2_0.5964"

dir = sorted(glob.glob(f"{fname}/*/mix.mid"))    # TODO: this is just for fast evaluation
# dir = dir[:1]
dir2 = [k.replace(f"{fname}/", "/data/slakh2100_flac_redux/test/").replace("/mix.mid", "/all_src_v2.mid") for k in dir]
fnames = zip(dir2, dir)


def func(item):
    fname1, fname2 = item
    # print(fname1, fname2)

    # 1. Perceiver style onset F1
    # return compute_transcription_metrics(fname1, fname2)

    # 2. Multi-instrument F1
    # ref_mid = pretty_midi.PrettyMIDI(fname1)
    # est_mid = pretty_midi.PrettyMIDI(fname2)
    # results = {}
    # for granularity in ["flat", "midi_class", "full"]:
    #     dic = mt3_program_aware_note_scores(
    #         ref_mid, est_mid, granularity
    #     )
    #     results.update(dic)
    
    # 3. Multi-instrument F1 V2
    results = {}
    for granularity in ["flat", "midi_class", "full"]:
        dic = mt3_program_aware_note_scores_v2(
            fname1, fname2, granularity, offset=True       # NOTE: offset should be true in MT3
        )
        results.update(dic)
    
    return results


pbar = tqdm(total=len(dir))
scores = collections.defaultdict(list)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Start the load operations and mark each future with its URL
    future_to_fname = {executor.submit(func, fname): fname for fname in fnames}
    for future in concurrent.futures.as_completed(future_to_fname):
        try:
            fname = future_to_fname[future]
            dic = future.result()
            for item in dic:
                scores[item].append(dic[item])
            pbar.update()
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            

# ref_sum, est_sum = [], []
# flat, midi_class, full, new, inst_num_ref, inst_num_est = 0, 0, 0, 0, 0, 0
# scores = collections.defaultdict(list)
# for elem in tqdm(dir):
#     name = elem.split("/")[-2]
    # ref_mid = pretty_midi.PrettyMIDI(elem)
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("loops_out", name, "mix.mid"))
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("babyslakh_16k_out", name, "mix.mid"))
    # est_mid = pretty_midi.PrettyMIDI(os.path.join("slakh2100_flac_redux_test_16k_orig", name, "mix.mid"))

    # for granularity in ["flat", "midi_class", "full"]:
    #     for score_name, score in mt3_program_aware_note_scores(
    #         ref_mid, 
    #         est_mid, 
    #         # os.path.join("loops_out", name, "mix.mid"),
    #         # f"out/{name}/mix.mid",
    #         granularity_type=granularity
    #     ).items():
    #         scores[score_name].append(score)

    # for score_name, score in compute_transcription_metrics(
    #     elem, 
    #     os.path.join("slakh2100_flac_redux_test_16k_orig", name, "mix.mid"), 
    # ).items():
    #     scores[score_name].append(score)
    
    # for granularity in ["flat", "midi_class", "full"]:
    #     for score_name, score in mt3_program_aware_note_scores_v2(
    #         elem, 
    #         os.path.join("slakh2100_flac_redux_test_16k_orig", name, "mix.mid"), 
    #         # os.path.join("loops_out", name, "mix.mid"),
    #         # f"out/{name}/mix.mid",
    #         granularity_type=granularity
    #     ).items():
    #         scores[score_name].append(score)

mean_scores = {k: np.mean(v) for k, v in scores.items()}
for key in sorted(list(mean_scores)):
    print("{}: {:.4}".format(key, mean_scores[key]))

