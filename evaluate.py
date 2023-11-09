"""
Multi-track transcription evaluation script for Slakh dataset.
"""
import mir_eval
import glob
import pretty_midi
import numpy as np
import librosa
import note_seq
import collections
import concurrent.futures
import traceback
from tqdm import tqdm


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


def mt3_program_aware_note_scores(
    fname1, fname2, granularity_type
):
    """
    Edited version of MT3's program aware precision/recall/F1 score.
    We follow Perceiver's evaluation approach which takes only onset and program into account.
    Using MIDIs transcribed from MT3, we managed to get similar results as Perceiver, which is 0.75 for onset F1.
    """
    ref_mid = pretty_midi.PrettyMIDI(fname1)
    est_mid = pretty_midi.PrettyMIDI(fname2)

    res = dict()
    ref_ns = note_seq.midi_file_to_note_sequence(fname1)
    est_ns = note_seq.midi_file_to_note_sequence(fname2)
    
    # NOTE: We don't need to remove drums and process separately as in MT3
    # as we consider onset only for all instruments.
    # def remove_drums(ns):
    #   ns_drumless = note_seq.NoteSequence()
    #   ns_drumless.CopyFrom(ns)
    #   del ns_drumless.notes[:]
    #   ns_drumless.notes.extend([note for note in ns.notes if not note.is_drum])
    #   return ns_drumless

    # est_ns_drumless = remove_drums(est_ns)
    # ref_ns_drumless = remove_drums(ref_ns)

    est_tracks = [est_ns]
    ref_tracks = [ref_ns]
    use_track_offsets = [False]
    use_track_velocities = [False]
    track_instrument_names = ['']

    # this part calculates instrument-agnostic onset F1 score
    # it is the same as: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L255
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

    # group notes by program number
    ref_inst_to_notes_mapping = {}
    est_inst_to_notes_mapping = {}

    # this part calculates multi-instrument onset F1 score
    # based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L36
    
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
    
    # this part is based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L82
    program_and_is_drum_tuples = set(ref_inst_to_notes_mapping.keys()) | set(est_inst_to_notes_mapping.keys())
    drum_precision_sum = 0.0
    drum_precision_count = 0
    drum_recall_sum = 0.0
    drum_recall_count = 0

    nondrum_precision_sum = 0.0
    nondrum_precision_count = 0
    nondrum_recall_sum = 0.0
    nondrum_recall_count = 0

    program_f1 = {}
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
        precision, recall, f_measure, unused_avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(**args))
        
        # print(f"program={program} is_drum={is_drum} est={est_pitches.shape[0]} ref={ref_pitches.shape[0]}")
        # print(f"precision={precision} recall={recall} f_measure={f_measure}")
        # print(f"est_intervals={len(est_intervals)} ref_intervals={len(ref_intervals)}")
        # print("======")
        
        if granularity_type == "midi_class":
            if is_drum:
                program_f1[-1] = f_measure
            else:
                program_f1[program] = f_measure

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

    # print(f"precision_sum={precision_sum} precision_count={precision_count}")
    # print(f"recall_sum={recall_sum} recall_count={recall_count}")

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
        f'Onset + program precision ({granularity_type})': precision,
        f'Onset + program recall ({granularity_type})': recall,
        f'Onset + program F1 ({granularity_type})': f_measure,
        # f'Drum onset precision ({granularity_type})': drum_precision,
        # f'Drum onset recall ({granularity_type})': drum_recall,
        # f'Drum onset F1 ({granularity_type})': drum_f_measure,
        # f'Nondrum onset + program precision ({granularity_type})':
        #     nondrum_precision,
        # f'Nondrum onset + program recall ({granularity_type})':
        #     nondrum_recall,
        # f'Nondrum onset + program F1 ({granularity_type})':
        #     nondrum_f_measure
        "F1 by program": program_f1
    })
    return res


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
    

def evaluate_main(
    dataset_name,   # "Slakh" or "ComMU"
    test_midi_dir,
    ground_truth_midi_dir,
    enable_instrument_eval=False
):
    if dataset_name == "Slakh":
        dir = sorted(glob.glob(f"{test_midi_dir}/*/mix.mid"))
        dir2 = [k.replace(test_midi_dir, ground_truth_midi_dir).replace("/mix.mid", "/all_src_v2.mid") for k in dir]
        fnames = zip(dir2, dir)
    
    elif dataset_name == "ComMU":
        dir = sorted(glob.glob(f"{test_midi_dir}/*.mid"))
        dir2 = [k.replace(test_midi_dir, ground_truth_midi_dir).replace("_16k.mid", ".mid") for k in dir]
        fnames = zip(dir2, dir)
    
    elif dataset_name == "NSynth":
        dir = sorted(glob.glob(f"{test_midi_dir}/*.mid"))
        dir2 = [k.replace(test_midi_dir, ground_truth_midi_dir).replace("_16k.mid", ".mid") for k in dir]
        fnames = zip(dir2, dir)

    else:
        raise ValueError("dataset_name must be either Slakh or ComMU")


    def func(item):
        fname1, fname2 = item

        results = {}
        for granularity in ["flat", "full", "midi_class"]:
            # print("\ngranularity:", granularity)
            dic = mt3_program_aware_note_scores(
                fname1, fname2, granularity
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
                

    mean_scores = {k: np.mean(v) for k, v in scores.items() if k != "F1 by program"}
    for key in sorted(list(mean_scores)):
        print("{}: {:.4}".format(key, mean_scores[key]))

    if enable_instrument_eval:
        print("====")
        program_f1_dict = {}
        for item in scores["F1 by program"]:
            for key in item:
                if key not in program_f1_dict:
                    program_f1_dict[key] = []
                program_f1_dict[key].append(item[key])

        d = {
            -1: "Drums",
            0: "Piano",
            1: "Chromatic Percussion",
            2: "Organ",
            3: "Guitar",
            4: "Bass",
            5: "Strings",
            6: "Ensemble",
            7: "Brass",
            8: "Reed",
            9: "Pipe",
            10: "Synth Lead",
            11: "Synth Pad",
            12: "Synth Effects",
        }
        program_f1_dict = {k: np.mean(np.array(v)) for k, v in program_f1_dict.items()}
        for key in d:
            if key == -1:
                print("{}: {:.4}".format(d[key], program_f1_dict[key]))
            elif key * 8 in program_f1_dict:
                print("{}: {:.4}".format(d[key], program_f1_dict[key * 8]))


if __name__ == "__main__":
    evaluate_main(
        "Slakh",
        test_midi_dir="outputs/2023-11-07/22-10-36/commu_mt3_on_slakh/",
        ground_truth_midi_dir="/data/slakh2100_flac_redux/test/",
    )