## MT3: Multi-Task Multitrack Music Transcription - Pytorch

Implementation of MT3 in pytorch currently support only inference using official pretrained weight released in https://github.com/magenta/mt3.

## Usage

```python
from inference import InferenceHandler

handler = InferenceHandler('./pretrained')
handler.inference('music.mp3')
```

## Citations

```bibtex
@article{gardner2021mt3,
  title={MT3: Multi-Task Multitrack Music Transcription},
  author={Gardner, Josh and Simon, Ian and Manilow, Ethan and Hawthorne, Curtis and Engel, Jesse},
  journal={arXiv preprint arXiv:2111.03017},
  year={2021}
}
```