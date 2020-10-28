# EEND (End-to-End Neural Diarization)

EEND (End-to-End Neural Diarization) is a neural-network-based speaker diarization method.
- BLSTM EEND (INTERSPEECH 2019)
  - https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2899.html
- Self-attentive EEND (ASRU 2019)
  - https://ieeexplore.ieee.org/abstract/document/9003959/

The EEND extension for various number of speakers is also provided in this repository.
- Self-attentive EEND with encoder-decoder based attractors
  - https://arxiv.org/abs/2005.09921

## Install tools
### Requirements
 - NVIDIA CUDA GPU
 - CUDA Toolkit (8.0 <= version <= 10.1)

### Install kaldi and python environment
```bash
cd tools
make
```
- This command builds kaldi at `tools/kaldi`
  - if you want to use pre-build kaldi
    ```bash
    cd tools
    make KALDI=<existing_kaldi_root>
    ```
    This option make a symlink at `tools/kaldi`
- This command extracts miniconda3 at `tools/miniconda3`, and creates conda envirionment named 'eend'
- Then, installs Chainer and cupy into 'eend' environment
  - use CUDA in `/usr/local/cuda/`
    - if you need to specify your CUDA path
      ```bash
      cd tools
      make CUDA_PATH=/your/path/to/cuda-8.0
      ```
      This command installs cupy-cudaXX according to your CUDA version.
      See https://docs-cupy.chainer.org/en/stable/install.html#install-cupy

## Test recipe (mini_librispeech)
### Configuration
- Modify `egs/mini_librispeech/v1/cmd.sh` according to your job schedular.
If you use your local machine, use "run.pl".
If you use Grid Engine, use "queue.pl"
If you use SLURM, use "slurm.pl".
For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
### Data preparation
```bash
cd egs/mini_librispeech/v1
./run_prepare_shared.sh
```
### Run training, inference, and scoring
```bash
./run.sh
```
- If you use encoder-decoder based attractors [3], modify `run.sh` to use `config/eda/{train,infer}.yaml`
- See `RESULT.md` and compare with your result.

## CALLHOME two-speaker experiment
### Configuraition
- Modify `egs/callhome/v1/cmd.sh` according to your job schedular.
If you use your local machine, use "run.pl".
If you use Grid Engine, use "queue.pl"
If you use SLURM, use "slurm.pl".
For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
- Modify `egs/callhome/v1/run_prepare_shared.sh` according to storage paths of your corpora.

### Data preparation
```bash
cd egs/callhome/v1
./run_prepare_shared.sh
# If you want to conduct 1-4 speaker experiments, run below.
# You also have to set paths to your corpora properly.
./run_prepare_shared_eda.sh
```
### Self-attention-based model using 2-speaker mixtures
```bash
./run.sh
```
### BLSTM-based model using 2-speaker mixtures
```bash
local/run_blstm.sh
```
### Self-attention-based model with EDA using 1-4-speaker mixtures
```bash
./run_eda.sh
```

## References
[1] Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Kenji Nagamatsu, Shinji Watanabe, "
End-to-End Neural Speaker Diarization with Permutation-free Objectives," Proc. Interspeech, pp. 4300-4304, 2019

[2] Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Yawen Xue, Kenji Nagamatsu, Shinji Watanabe, "
End-to-End Neural Speaker Diarization with Self-attention," Proc. ASRU, pp. 296-303, 2019

[3] Shota Horiguchi, Yusuke Fujita, Shinji Watanabe, Yawen Xue, Kenji Nagamatsu, "
End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors," Proc. INTERSPEECH, 2020



## Citation
```
@inproceedings{Fujita2019Interspeech,
 author={Yusuke Fujita and Naoyuki Kanda and Shota Horiguchi and Kenji Nagamatsu and Shinji Watanabe},
 title={{End-to-End Neural Speaker Diarization with Permutation-free Objectives}},
 booktitle={Interspeech},
 pages={4300--4304}
 year=2019
}
```
