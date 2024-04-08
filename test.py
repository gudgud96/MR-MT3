import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
from inference import InferenceHandler
import glob
import os
from tqdm import tqdm
import librosa
import hydra
import numpy as np
from evaluate import evaluate_main
from tasks.mt3_net import MT3Net
from tasks.mt3_net_segmem import MT3NetSegMem


def get_scores(
    model,
    eval_audio_dir=None,
    mel_norm=True,
    eval_dataset="Slakh",
    exp_tag_name="test_midis",
    ground_truth_midi_dir=None,
    verbose=True,
    contiguous_inference=False,
    use_tf_spectral_ops=False,
    batch_size=8,
    max_length=1024
):  
    handler = InferenceHandler(
        model=model,
        device=torch.device('cuda'),
        mel_norm=mel_norm,
        contiguous_inference=contiguous_inference,
        use_tf_spectral_ops=use_tf_spectral_ops
    )

    def func(fname):
        audio, _ = librosa.load(fname, sr=16000)
        if eval_dataset == "NSynth":
            audio = np.pad(audio, (int(0.05 * 16000), 0), "constant", constant_values=0)
        return audio

    if verbose:
        print("Total songs:", len(eval_audio_dir))

    for fname in tqdm(eval_audio_dir):
        audio = func(fname)
        
        if eval_dataset == "Slakh":
            name = fname.split("/")[-2]
            outpath = os.path.join(exp_tag_name, name, "mix.mid")
        elif eval_dataset == "ComMU" or eval_dataset == "NSynth":
            name = fname.split("/")[-1]
            outpath = os.path.join(exp_tag_name,  name.replace(".wav", ".mid"))
        else:
            raise ValueError("Invalid dataset name.")

        handler.inference(
            audio=audio, 
            audio_path=fname, 
            outpath=outpath,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose
        )
    
    if verbose:
        print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    scores = evaluate_main(
        dataset_name=eval_dataset,
        test_midi_dir=os.path.join(current_dir, exp_tag_name),
        ground_truth_midi_dir=ground_truth_midi_dir,
    )

    if verbose:
        for key in sorted(list(scores)):
            print("{}: {:.4}".format(key, scores[key]))

    return scores


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    assert cfg.path
    assert cfg.path.endswith(".pt") or \
            cfg.path.endswith("pth") or \
            cfg.path.endswith("ckpt"), "Only .pt, .pth, .ckpt files are supported."
    assert cfg.eval.exp_tag_name
    assert cfg.eval.audio_dir

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    if cfg.path.endswith(".ckpt"):
        # load lightning module from checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_) 
        print("torch.cuda.device_count():", torch.cuda.device_count())
        pl = model_cls.load_from_checkpoint(
            cfg.path,
            config=cfg.model.config,
            optim_cfg=cfg.optim,
        )
        model = pl.model
    else:
        # load weights for nn.Module
        model = pl.model
        if cfg.eval.load_weights_strict is not None:
            model.load_state_dict(torch.load(cfg.path), strict=cfg.eval.load_weights_strict)
        else:
            model.load_state_dict(torch.load(cfg.path), strict=False)

    model.eval()
    # if torch.cuda.is_available():
    #     model.cuda()

    dir = sorted(glob.glob(cfg.eval.audio_dir))
    if cfg.eval.eval_dataset == "NSynth":
        # NOTE: skip vocals and mallets. Both Slakh and ComMU dataset does not have vocals, and ComMU does not have mallets.
        dir = [d for d in dir if "vocal" not in d and "mallet" not in d]
    if cfg.eval.eval_first_n_examples:
        dir = dir[:cfg.eval.eval_first_n_examples]

    mel_norm = False if "pretrained/mt3.pth" in cfg.path else True
    ground_truth_midi_dir = cfg.eval.midi_dir if cfg.eval.midi_dir else cfg.dataset.test.root_dir

    get_scores(
        model,
        eval_audio_dir=dir,
        mel_norm=mel_norm,
        eval_dataset=cfg.eval.eval_dataset,
        exp_tag_name=cfg.eval.exp_tag_name,
        ground_truth_midi_dir=ground_truth_midi_dir,
        contiguous_inference=cfg.eval.contiguous_inference,
        use_tf_spectral_ops=cfg.eval.use_tf_spectral_ops,
        batch_size=cfg.eval.batch_size,
    )


if __name__ == "__main__":   
    main()