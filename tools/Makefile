# Makefile
# Copyright 2018-2019 Hitachi, Ltd. (author: Yusuke Fujita)
#
# install tools

# If you want to use prebuild kaldi, make KALDI=<path/to/kaldi_root>
KALDI :=
# Specify cuda root path installed in your environment
CUDA_PATH := /usr/local/cuda
CUDA_VERSION := $(shell $(CUDA_PATH)/bin/nvcc --version | tail -n1 | awk '{print substr($$5,0,length($$5)-1)}')

all: kaldi miniconda3/envs/eend/bin env.sh

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(abspath $(KALDI)) kaldi
else
kaldi:
	git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE)
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

miniconda3.sh:
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $@

miniconda3: miniconda3.sh
	# -b: non-interactive install
	# -p: installed directory
	bash miniconda3.sh -b -p miniconda3

# virtual environment of python
miniconda3/envs/eend/bin: miniconda3
	miniconda3/bin/conda update -y conda
	miniconda3/bin/conda env create -f environment.yml
	miniconda3/envs/eend/bin/pip install cupy-cuda$(subst .,,$(CUDA_VERSION))==6.2.0 chainer==6.2.0
update:
	miniconda3/bin/conda env update -f environment.yml

env.sh: miniconda3/envs/eend/bin
	cp env.sh.in env.sh
	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH" >> env.sh
