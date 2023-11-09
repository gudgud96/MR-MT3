def get_inst_dict():
    inst_dict = {
        "accordion": 22,
        "acoustic_bass": 33,
        "acoustic_guitar": 25,
        "acoustic_piano": 1,
        "bassoon": 71,
        "bell": 15,
        "brass_ensemble": 62,
        "celesta": 9,
        "choir": 53,
        "clarinet": 72,
        "electric_bass": 34,
        "electric_guitar_clean": 28,
        "electric_guitar_distortion": 31,
        "electric_piano": 5,
        "flute": 74,
        "glockenspiel": 10,
        "harp": 47,
        "horn": 61,                 # french horn
        "marimba": 13,
        "nylon_guitar": 25,
        "oboe": 69,
        "orgel": 17,
        "string_cello": 43,
        "string_double_bass": 44,
        "string_ensemble": 49,
        "string_viola": 42,
        "string_violin": 41,
        "synth_bass": 39,
        "synth_bass_wobble": 39,    # no wobble sound in soundbank
        "synth_bell": 15,           # no difference with bell
        "synth_pad": 89,
        "synth_pluck": 83,          # no pluck, so i choose "calliope lead"
        "synth_voice": 86,
        "timpani": 48,
        "trombone": 58,
        "tuba": 59,
        "vibraphone": 12,
        "xylophone": 14
    }
    
    for k, v in inst_dict.items():
        inst_dict[k] = v - 1        # program starts from 0
    
    return inst_dict