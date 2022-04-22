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
@unknown{unknown,
author = {Gardner, Josh and Simon, Ian and Manilow, Ethan and Hawthorne, Curtis and Engel, Jesse},
year = {2021},
month = {11},
pages = {},
title = {MT3: Multi-Task Multitrack Music Transcription}
}
```