import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from inference import InferenceHandler
import torch
import glob
import os
from tqdm import tqdm
import librosa
import hydra
from evaluate import evaluate_main


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # convert .ckpt to .pth
    assert cfg.path
    # assert cfg.path.endswith(".pt") or cfg.path.endswith("pth"), "Only .pt / .pth files are supported."
    assert cfg.eval.exp_tag_name
    assert cfg.eval.audio_dir

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    model = pl.model
    print(f"Loading weights from: {cfg.path}")
    if cfg.path.endswith(".ckpt"):
        print("============Converting .ckpt to .pth...")
        state_dict = torch.load(cfg.path)['state_dict']
        dic = {}
        for key in state_dict:
            if "model." in key:
                dic[key.replace("model.", "")] = state_dict[key]
            elif "criterion." in key:
                pass                
            else:
                dic[key] = state_dict[key]   
        model.load_state_dict(dic)
    else:
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

    for fname in tqdm(dir):
        audio = func(fname)
        
        if cfg.eval.eval_dataset == "Slakh":
            name = fname.split("/")[-2]
            outpath = os.path.join(exp_tag_name, name, "mix.mid")
        elif cfg.eval.eval_dataset == "ComMU":
            name = fname.split("/")[-1]
            outpath = os.path.join(exp_tag_name,  name.replace(".wav", ".mid"))
        else:
            raise ValueError("Invalid dataset name.")

        handler.inference(
            audio, 
            fname, 
            outpath=outpath,
            batch_size=8,        # changing this might affect results of sequential-inference models (e.g. XL).
            max_length=256
        )
    
    print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    evaluate_main(
        dataset_name=cfg.eval.eval_dataset,
        test_midi_dir=os.path.join(current_dir, cfg.eval.exp_tag_name),
        ground_truth_midi_dir=cfg.dataset.test.root_dir,
    )


if __name__ == "__main__":   
    main()