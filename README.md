# EEND (End-to-End Neural Diarization)

EEND (End-to-End Neural Diarization) is a neural-network-based speaker diarization method.

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
- See `RESULT.md` and compare with your result.

## CALLHOME two-speaker experiment
### Configuraition
- Modify `egs/callhome/v1/cmd.sh` according to your job schedular.
If you use your local machine, use "run.pl".
If you use Grid Engine, use "queue.pl"
If you use SLURM, use "slurm.pl".
For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
- Modify `egs/callhome/v1/run_prepare_shared.sh` according to storage paths of your copora.

### Data preparation
```bash
cd egs/callhome/v1
./run_prepare_shared.sh
```
### Self-attention-based model (latest configuration)
```bash
./run.sh
```
### BLSTM-based model (old configuration)
```bash
local/run_blstm.sh
```
## Citation
```
@inproceedings{Fujita2019Interspeech,
 author={Yusuke Fujita and Naoyuki Kanda and Shota Horiguchi and Kenji Nagamatsu and Shinji Watanabe},
 title={End-to-end Neural Speaker Diarization with Permutation-free Objectives},
 booktitle={Interspeech},
 year=2019
}
```
