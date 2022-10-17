# PlayItBack
### Code implementation for:
Play It Back: Iterative Attention for Audio Recognition

- <a href="https://alexandrosstergiou.github.io/project_pages/TemPr/index.html">[Project page]</a>

- <a href="http://arxiv.org/abs/2204.13340">[ArXiv paper]</a>

![supported versions](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue/?style=flat&logo=pytorch&color=informational)
![GitHub license](https://img.shields.io/badge/license-Apache2.0-lightgrey)


## Abstract
A key function of auditory cognition is the association of characteristic sounds with their corresponding semantics over time.
Humans attempting to discriminate between fine-grained audio categories, often replay the same discriminative sounds to increase their prediction confidence.
We propose an end-to-end attention-based architecture that through selective repetition attends over the most discriminative sounds across the audio sequence. Our model initially uses the full audio sequence and iteratively refines the temporal segments replayed based on slot attention. At each playback, the selected segments are replayed using a smaller hop length which represents higher resolution features within these segments. 
We show that our method can consistently achieve state-of-the-art performance across three audio-classification benchmarks: AudioSet, VGG-Sound, and EPIC-KITCHENS-100. 

<p align="center">
<img src="./figs/PlayItBack-PlayItBack.png" width="700" />
</p>


## Dependencies

Ensure that the following packages are installed in your machine:

  * [PyTorch](https://pytorch.org) 
  * [librosa](https://librosa.org)
  * [h5py](https://www.h5py.org)
  * [wandb](https://wandb.ai/site)
  * [fvcore](https://github.com/facebookresearch/fvcore/)
  * [simplejson](https://pypi.org/project/simplejson/)
  * [psutil](https://pypi.org/project/psutil/)
  * [tensorboard](https://www.tensorflow.org/tensorboard/) 

Audioset flac



## Datasets


## Usage


## Acknowledgement
This repository is built based on [auditory-slow-fast](https://github.com/ekazakos/auditory-slow-fast) and [PySlowFast](https://github.com/facebookresearch/SlowFast).


## Citation

```
@article{stergiou2022playitback,
title={Play It Back: Iterative Attention for Audio Recognition},
author={Stergiou, Alexandros and Damen, Dima},
journal={arXiv preprint},
year={2022}}
```

## License

MIT
