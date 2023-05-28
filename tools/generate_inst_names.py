from glob import glob
import yaml
import json

from contrib.preprocessor import _SLAKH_CLASS_PROGRAMS

_SLAKH_CLASS_PROGRAMS

def _find_inst_name(program_num):
    inst = None
    for i, (k, v) in enumerate(_SLAKH_CLASS_PROGRAMS.items()):
        if program_num >= v:
            inst = k
        else:
            break
    assert inst is not None
    return inst

def main(root_path):
    meta_paths = glob(f'{root_path}/**/metadata.yaml')
    for meta_path in meta_paths:
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
            inst_names_path = meta_path.replace('metadata.yaml', 'inst_names.json')
            inst_names = {}
            for k in metadata['stems'].keys():
                # print(k, metadata['stems'][k]['inst_class'])
                if(metadata['stems'][k].get('integrated_loudness', None) is not None):
                    inst_names[k] = _find_inst_name(metadata['stems'][k]['program_num'])
            with open(inst_names_path, 'w') as w:
                json.dump(inst_names, w)
    print('done')
            


if __name__ == '__main__':
    main('/mnt/2TB/dataset/midi/babyslakh_16k')
