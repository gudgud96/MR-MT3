import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from inference import InferenceHandler
import torch
import glob
import os
from tqdm import tqdm
import librosa
import hydra
import numpy as np
from evaluate import evaluate_main


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # convert .ckpt to .pth
    assert cfg.path
    assert cfg.path.endswith(".pt") or cfg.path.endswith("pth"), "Only .pt / .pth files are supported."
    assert cfg.eval.exp_tag_name
    assert cfg.eval.audio_dir

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    model = pl.model
    print(f"Loading weights from: {cfg.path}")
    model.load_state_dict(torch.load(cfg.path))
    model.eval()

    dir = sorted(glob.glob(cfg.eval.audio_dir))
    if cfg.eval.is_sanity_check:
        dir = dir[:10]

    mel_norm = False if "pretrained/mt3.pth" in cfg.path else True
    handler = InferenceHandler(
        model=model,
        device=torch.device('cuda'),
        mel_norm=mel_norm,
        contiguous_inference=False
    )

    def func(fname):
        audio, _ = librosa.load(fname, sr=16000)
        return audio

    print("Total songs:", len(dir))

    exp_tag_name = cfg.eval.exp_tag_name

    invalid_events_perc, dropped_events_perc, adjacent_program_events_perc = 0, 0, 0
    est_exc_dict = {}

    for fname in tqdm(dir):
        audio = func(fname)
        
        if cfg.eval.eval_dataset == "Slakh":
            name = fname.split("/")[-2]
            outpath = os.path.join(exp_tag_name, name, "mix.mid")
        elif cfg.eval.eval_dataset == "ComMU" or cfg.eval.eval_dataset == "NSynth":
            name = fname.split("/")[-1]
            outpath = os.path.join(exp_tag_name,  name.replace(".wav", ".mid"))
        else:
            raise ValueError("Invalid dataset name.")

        ret_dict = handler.inference(
            audio, 
            fname, 
            outpath=outpath,
            batch_size=8,        # changing this might affect results of sequential-inference models (e.g. XL).
            max_length=256
        )

        invalid_events_perc += ret_dict["invalid_events_perc"]
        dropped_events_perc += ret_dict["dropped_events_perc"]
        adjacent_program_events_perc += ret_dict["adjacent_program_events_perc"]
        for key in ret_dict["est_exc_dict"]:
            if key not in est_exc_dict:
                est_exc_dict[key] = ret_dict["est_exc_dict"][key]
            else:
                est_exc_dict[key] += ret_dict["est_exc_dict"][key]
    
    print("==========================================")
    print("|          Invalid events check          |")
    print("==========================================")
    print("Invalid events percentage:", invalid_events_perc / len(dir))
    print("Dropped events percentage:", dropped_events_perc / len(dir))
    print("Adjacent program events percentage:", adjacent_program_events_perc / len(dir))
    print()
    print("Invalid events breakdown:")
    for key in est_exc_dict:
        print('"' + key + '" - ', est_exc_dict[key])
    print("==========================================")

    
    print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    ground_truth_midi_dir = cfg.eval.midi_dir if cfg.eval.midi_dir is not None else cfg.dataset.test.root_dir
    evaluate_main(
        dataset_name=cfg.eval.eval_dataset,
        test_midi_dir=os.path.join(current_dir, cfg.eval.exp_tag_name),
        ground_truth_midi_dir=ground_truth_midi_dir,
    )


if __name__ == "__main__":   
    main()